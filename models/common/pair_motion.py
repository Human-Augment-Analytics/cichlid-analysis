"""Shared male/partner pair-motion features for frame and temporal models.

The feature definitions live in feature_engineering/pair_motion. This adapter
builds one video at a time, selects the same male/partner slots used by the
MS-TCN converter, canonicalizes directional features by fish role, and caches a
single row per original video frame. No labels are used by feature generation.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from feature_engineering.pair_motion.feature_groups import ALL_MODEL_FEATURES
from feature_engineering.pair_motion.features_motion import (
    build_motion_features,
    run_synthetic_checks,
    spot_check_features,
)
from feature_engineering.pair_motion.io_pose import PAIR_COLUMNS, TRACK_COLUMNS, assert_contract
from models.common.config import DATA_DIR, OUTPUTS
from models.intervals.mstcn_dlc2action.dlc_yolo_converter import dlc2action_bridge as br

VERSION = "male_partner_pair_motion_v2_persistence"
FEATURE_DIR = OUTPUTS / "frame_features"
PREFIX = "pm_"

ROLE_NAMES = {
    "dist_nose1_tailtip2_bl": "dist_nose_male_tailtip_partner_bl",
    "dist_nose2_tailtip1_bl": "dist_nose_partner_tailtip_male_bl",
    "dist_nose1_spine4of2_bl": "dist_nose_male_spine4_partner_bl",
    "dist_nose2_spine4of1_bl": "dist_nose_partner_spine4_male_bl",
    "speed_bl_s_1": "speed_male_bl_s",
    "speed_bl_s_2": "speed_partner_bl_s",
    "speed_diff_bl_s": "speed_male_minus_partner_bl_s",
    "facing_1": "facing_male",
    "facing_2": "facing_partner",
    "d_heading_1_rad_s_clean": "d_heading_male_rad_s_clean",
    "d_heading_2_rad_s_clean": "d_heading_partner_rad_s_clean",
    "bearing_1_sin": "bearing_male_sin",
    "bearing_1_cos": "bearing_male_cos",
    "bearing_2_sin": "bearing_partner_sin",
    "bearing_2_cos": "bearing_partner_cos",
}
ROLE_SWAPS = (
    ("dist_nose1_tailtip2_bl", "dist_nose2_tailtip1_bl"),
    ("dist_nose1_spine4of2_bl", "dist_nose2_spine4of1_bl"),
    ("speed_bl_s_1", "speed_bl_s_2"),
    ("facing_1", "facing_2"),
    ("d_heading_1_rad_s_clean", "d_heading_2_rad_s_clean"),
    ("bearing_1_sin", "bearing_2_sin"),
    ("bearing_1_cos", "bearing_2_cos"),
)
PAIR_MOTION_COLUMNS = [PREFIX + ROLE_NAMES.get(name, name) for name in ALL_MODEL_FEATURES]
FRAME_COLUMNS = [*PAIR_MOTION_COLUMNS, PREFIX + "pair_present"]


def _file_signature(path: Path) -> dict:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _signature(video: str, frame_count: int, data_dir: Path, slots_file: Path) -> dict:
    inputs = [data_dir / f"{video}.parquet", data_dir / f"{video}_pairs.parquet", slots_file]
    return {
        "version": VERSION,
        "video": video,
        "frames": int(frame_count),
        "inputs": {path.name: _file_signature(path) for path in inputs},
    }


def _read_inputs(video: str, data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    track_path = data_dir / f"{video}.parquet"
    pair_path = data_dir / f"{video}_pairs.parquet"
    missing = [str(path) for path in (track_path, pair_path) if not path.exists()]
    if missing:
        raise SystemExit(f"missing input file(s): {missing} (set CIRCLING_DATA_DIR)")
    assert_contract(track_path, "tracks")
    assert_contract(pair_path, "pairs")
    tracks = pd.read_parquet(track_path, columns=TRACK_COLUMNS)
    pairs = pd.read_parquet(pair_path, columns=PAIR_COLUMNS)
    if tracks.empty or pairs.empty:
        raise ValueError(f"{video}: pair-motion features need nonempty track and pair tables")
    if set(tracks["day_label"].unique()) != {video} or set(pairs["day_label"].unique()) != {video}:
        raise ValueError(f"{video}: day_label in the pose inputs must exactly match the video ID")
    return tracks, pairs


def _select_male_partner(features: pd.DataFrame, meta: dict, frame_count: int) -> pd.DataFrame:
    slots_file = Path(meta["slots_file"])
    uid = np.load(slots_file)["uid"]
    stride, first = int(meta.get("stride", 1)), int(meta.get("frame_min", 0))
    frames = features["FrameNum"].to_numpy(dtype=np.int64)
    rows = (frames - first) // stride
    valid = (frames >= first) & (rows >= 0) & (rows < len(uid)) & (frames < frame_count)
    features = features.loc[valid].copy()
    rows = rows[valid]
    male, partner = uid[rows, 0], uid[rows, 1]
    track_1 = features["TrackUID_1"].to_numpy(dtype=np.int64)
    track_2 = features["TrackUID_2"].to_numpy(dtype=np.int64)
    direct = (track_1 == male) & (track_2 == partner) & (male >= 0) & (partner >= 0)
    swapped = (track_1 == partner) & (track_2 == male) & (male >= 0) & (partner >= 0)
    selected = features.loc[direct | swapped].copy()
    selected["male_first"] = direct[direct | swapped]
    # The upstream contract ensures at most one row for a pair and frame. Keep
    # the first only as a guard against duplicate converted rows.
    selected = selected.drop_duplicates("FrameNum", keep="first")
    return selected


def _canonical_frame_matrix(selected: pd.DataFrame, frame_count: int) -> np.ndarray:
    values = np.full((frame_count, len(ALL_MODEL_FEATURES) + 1), np.nan, dtype=np.float32)
    if selected.empty:
        values[:, -1] = 0.0
        return values
    male_first = selected["male_first"].to_numpy(dtype=bool)
    swapped_values = {}
    for first, second in ROLE_SWAPS:
        a = selected[first].to_numpy(dtype=float)
        b = selected[second].to_numpy(dtype=float)
        swapped_values[first] = np.where(male_first, a, b)
        swapped_values[second] = np.where(male_first, b, a)
    rows = selected["FrameNum"].to_numpy(dtype=np.int64)
    for index, name in enumerate(ALL_MODEL_FEATURES):
        if name in swapped_values:
            column = swapped_values[name]
        else:
            column = selected[name].to_numpy(dtype=float)
        if name == "speed_diff_bl_s":
            column = np.where(male_first, column, -column)
        elif name == "rel_heading_sin":
            column = np.where(male_first, column, -column)
        values[rows, index] = column.astype(np.float32)
    values[rows, -1] = 1.0
    values[:, -1] = np.nan_to_num(values[:, -1], nan=0.0)
    return values


def build_pair_motion_table(video: str, meta: dict, frame_count: int, data_dir=DATA_DIR) -> pl.DataFrame:
    """Build uncached features; public callers should use pair_motion_table."""
    data_dir = Path(data_dir)
    tracks, pairs = _read_inputs(video, data_dir)
    print(f"Building engineered pair-motion features: {video}", flush=True)
    features, join_stats = build_motion_features(pairs, tracks)
    if join_stats["unmatched_fish_1"] or join_stats["unmatched_fish_2"]:
        raise ValueError(f"{video}: some pair rows did not join to both fish tracks: {join_stats}")
    spot_check_features(features)
    selected = _select_male_partner(features, meta, frame_count)
    matrix = _canonical_frame_matrix(selected, frame_count)
    table = pl.DataFrame({
        "FrameNum": np.arange(frame_count, dtype=np.int64),
        **{name: matrix[:, index] for index, name in enumerate(FRAME_COLUMNS)},
    })
    coverage = float(selected["FrameNum"].nunique() / frame_count)
    print(f"{video}: {len(FRAME_COLUMNS) - 1} pair-motion features; "
          f"male/partner pair present on {coverage:.1%} of frames", flush=True)
    return table


def pair_motion_cache_file(workdir, video: str, meta: dict, frame_count: int, data_dir=DATA_DIR) -> Path:
    """Return the full-frame Parquet cache, rebuilding when any input changes."""
    data_dir = Path(data_dir)
    slots_file = Path(meta.get("slots_file", ""))
    if not slots_file.is_file():
        raise SystemExit(f"{video}: {slots_file} not found; run pose conversion before pair-motion features")
    signature = _signature(video, frame_count, data_dir, slots_file)
    path = FEATURE_DIR / f"{video}_pairmotion.parquet"
    info = path.with_suffix(".json")
    if path.exists() and info.exists() and json.loads(info.read_text()) == signature:
        return path
    table = build_pair_motion_table(video, meta, frame_count, data_dir)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    table.write_parquet(path)
    info.write_text(json.dumps(signature, indent=2) + "\n")
    return path


def pair_motion_table(workdir, video: str, meta: dict, frame_count: int, data_dir=DATA_DIR) -> pl.DataFrame:
    """Load the cached full-frame pair-motion table."""
    path = pair_motion_cache_file(workdir, video, meta, frame_count, data_dir)
    return pl.read_parquet(path)


def pair_motion_feature_file(workdir, video: str, meta: dict, frame_count: int, data_dir=DATA_DIR) -> Path:
    """Write the stride-aligned, median-imputed DLC2Action feature file."""
    workdir = Path(workdir)
    table = pair_motion_table(workdir, video, meta, frame_count, data_dir)
    stride, first = int(meta.get("stride", 1)), int(meta.get("frame_min", 0))
    rows = np.arange(int(meta["n_frames"]), dtype=np.int64) * stride + first
    sampled = pl.DataFrame({"FrameNum": rows}).join(table, on="FrameNum", how="left", validate="1:1")
    values = sampled.select(FRAME_COLUMNS).to_numpy().astype(np.float32)
    for index in range(values.shape[1] - 1):
        column = values[:, index]
        good = np.isfinite(column)
        column[~good] = np.median(column[good]) if good.any() else 0.0
    values[:, -1] = np.nan_to_num(values[:, -1], nan=0.0)
    data = workdir / "data"
    data.mkdir(parents=True, exist_ok=True)
    path = data / f"{video}_pairmotion.pickle"
    with path.open("wb") as handle:
        pickle.dump({br.PAIR_CLIP: values}, handle)
    (data / "_pairmotion_info.json").write_text(json.dumps({
        "feature_names": FRAME_COLUMNS,
        "version": VERSION,
        "imputation": "per-video median; pair_present is retained and never imputed",
    }, indent=2) + "\n")
    return path


def combine_motion_embedding(workdir, video: str, meta: dict, motion: str, embedding: str) -> Path:
    """Replace the original pair block in an embedding file with pair-motion."""
    data = Path(workdir) / "data"
    motion_path = data / f"{video}_{motion}.pickle"
    embedding_path = data / f"{video}_{embedding}.pickle"
    target = data / f"{video}_{motion}{embedding.removeprefix('pair')}.pickle"
    # pairmotion + pairemb -> pairmotionemb; pairmotion + pairvmae -> pairmotionvmae.
    info_path = data / f"_{target.stem.removeprefix(video + '_')}_info.json"
    sources = {path.name: _file_signature(path) for path in (motion_path, embedding_path)}
    if target.exists() and info_path.exists():
        info = json.loads(info_path.read_text())
        if info.get("sources") == sources:
            return target
    with motion_path.open("rb") as handle:
        motion_values = np.asarray(pickle.load(handle)[br.PAIR_CLIP], dtype=np.float32)
    with embedding_path.open("rb") as handle:
        embedding_values = np.asarray(pickle.load(handle)[br.PAIR_CLIP], dtype=np.float32)
    if len(motion_values) != len(embedding_values):
        raise ValueError(f"{video}: pair-motion and embedding row counts differ")
    n_pair = len((meta.get("pair_stats") or {}).get("pair_features") or [])
    if not n_pair:
        raise ValueError(f"{video}: original pair-feature metadata is missing")
    motion_info = json.loads((data / "_pairmotion_info.json").read_text())
    embedding_info_path = data / f"_{embedding}_info.json"
    embedding_info = json.loads(embedding_info_path.read_text()) if embedding_info_path.exists() else {}
    all_embedding_names = embedding_info.get("feature_names") or [
        f"col{index}" for index in range(embedding_values.shape[1])]
    names = list(motion_info["feature_names"]) + list(all_embedding_names[n_pair:])
    values = np.column_stack([motion_values, embedding_values[:, n_pair:]]).astype(np.float32)
    with target.open("wb") as handle:
        pickle.dump({br.PAIR_CLIP: values}, handle)
    info_path.write_text(json.dumps({
        "feature_names": names,
        "sources": sources,
        "motion": motion,
        "embedding": embedding,
    }, indent=2) + "\n")
    return target


def check_feature_definitions() -> None:
    """Run the feature package's synthetic checks on demand."""
    run_synthetic_checks()
    if len(PAIR_MOTION_COLUMNS) != len(set(PAIR_MOTION_COLUMNS)):
        raise AssertionError("Role-canonical pair-motion feature names are not unique")

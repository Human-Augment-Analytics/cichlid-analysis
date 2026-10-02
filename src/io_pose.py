"""Load validated pose tracks and pair tables, and check their metadata contract."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

DEFAULT_DATA_DIR = Path.home() / "Downloads"

TRACK_FILES = ("0028_vid.parquet", "0031_vid.parquet")
PAIR_FILES = ("0028_vid_pairs.parquet", "0031_vid_pairs.parquet")

EXPECTED_FPS = "30.0"
EXPECTED_Y_AXIS = "down"
EXPECTED_COORDINATE_SPACE = "raw_pixels"
EXPECTED_UNITS = "pixels"
EXPECTED_PAIR_ORDERING = "TrackUID_1 < TrackUID_2"
EXPECTED_GAP_TOLERANCE_S = "0.2"

TRACK_COLUMNS = [
    "project_id",
    "day_index",
    "day_label",
    "FrameNum",
    "Time_s",
    "TrackID",
    "TrackUID",
    "Sex",
    "Nose_x",
    "Nose_y",
    "TailTip_x",
    "TailTip_y",
    "body_length_px",
    "speed_bl_s",
    "qc_n_missing_kp",
    "is_track_start",
    "is_track_end",
]

PAIR_COLUMNS = [
    "project_id",
    "day_index",
    "day_label",
    "Time_s",
    "FrameNum",
    "TrackID_1",
    "TrackID_2",
    "TrackUID_1",
    "TrackUID_2",
    "centroid_distance_px",
    "dist_nose1_nose2_px",
    "dist_nose1_tailtip2_px",
    "dist_nose2_tailtip1_px",
    "dist_nose1_spine4of2_px",
    "dist_nose2_spine4of1_px",
    "orbital_rad",
    "rel_heading_rad",
    "bearing_1_rad",
    "bearing_2_rad",
    "sex_1",
    "sex_2",
    "pair_sex",
    "qc_n_missing_kp_1",
    "qc_n_missing_kp_2",
    "pair_episode_id",
    "dt_frames",
]

TRACK_KEYS = ["project_id", "day_label", "FrameNum", "TrackUID"]


def file_metadata(path: Path) -> dict[str, str]:
    parquet = pq.ParquetFile(path)
    raw = parquet.metadata.metadata or {}
    decoded: dict[str, str] = {}
    for key, value in raw.items():
        name = key.decode() if isinstance(key, bytes) else str(key)
        if name.startswith("ARROW") or name.startswith("pandas"):
            continue
        decoded[name] = value.decode() if isinstance(value, bytes) else str(value)
    return decoded


def assert_contract(path: Path, kind: str) -> dict[str, str]:
    """Raise if a pose file does not match the validated-track contract."""
    meta = file_metadata(path)
    checks = {
        "fps": EXPECTED_FPS,
        "y_axis": EXPECTED_Y_AXIS,
        "coordinate_space": EXPECTED_COORDINATE_SPACE,
        "units": EXPECTED_UNITS,
    }
    if kind == "pairs":
        checks["pair_ordering"] = EXPECTED_PAIR_ORDERING
        checks["gap_tolerance_s"] = EXPECTED_GAP_TOLERANCE_S
        checks["table"] = "pairs"
    mismatches = [
        f"{path.name}: {key}={meta.get(key)!r}, expected {expected!r}"
        for key, expected in checks.items()
        if meta.get(key) != expected
    ]
    if mismatches:
        raise ValueError("Pose metadata contract failed:\n" + "\n".join(mismatches))
    return meta


def load_pose(data_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, str]]]:
    """Load both videos' track and pair tables.

    Returns tracks, pairs, and the per-file metadata dictionaries.
    """
    root = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    metadata: list[dict[str, str]] = []
    tracks: list[pd.DataFrame] = []
    pairs: list[pd.DataFrame] = []

    for name in TRACK_FILES:
        path = root / name
        metadata.append({"file": name, "kind": "tracks", **assert_contract(path, "tracks")})
        tracks.append(pd.read_parquet(path, columns=TRACK_COLUMNS))

    for name in PAIR_FILES:
        path = root / name
        metadata.append({"file": name, "kind": "pairs", **assert_contract(path, "pairs")})
        pairs.append(pd.read_parquet(path, columns=PAIR_COLUMNS))

    track_df = pd.concat(tracks, ignore_index=True)
    pair_df = pd.concat(pairs, ignore_index=True)
    duplicate_tracks = int(track_df.duplicated(TRACK_KEYS).sum())
    if duplicate_tracks:
        raise ValueError(f"Track table has {duplicate_tracks} duplicate frame keys")
    duplicate_pairs = int(
        pair_df.duplicated(
            ["project_id", "day_label", "FrameNum", "TrackUID_1", "TrackUID_2"]
        ).sum()
    )
    if duplicate_pairs:
        raise ValueError(f"Pair table has {duplicate_pairs} duplicate frame keys")
    return track_df, pair_df, metadata

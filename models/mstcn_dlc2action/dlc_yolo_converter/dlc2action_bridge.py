import copy
import json
import logging
import math
import pickle
import random
import re
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:
    from .convert import BODYPARTS as _CONVERT_BODYPARTS
except ImportError:
    try:
        from convert import BODYPARTS as _CONVERT_BODYPARTS
        _CONVERT_BODYPARTS = None

logger = logging.getLogger("dlc_yolo_converter.dlc2action")

FALLBACK_BODYPARTS = ["Nose", "LeftEye", "RightEye", "Head", "Spine1",
                      "Spine2", "Spine3", "Spine4", "Peduncle", "TailTip"]
DEFAULT_BODYPARTS = list(_CONVERT_BODYPARTS or FALLBACK_BODYPARTS)

N_INDIVIDUALS = 2
PAIR_CLIP = "ind0+ind1"
ERROR_CLASS = "unknown"  
SUPPORTED_DLC2ACTION = ("1.0",)

ANN_COL_BEHAVIOR = "behavior"
ANN_COL_START = "start"
ANN_COL_END = "end"

_GT_ALIASES = {
    "video": {"video", "video_id", "videoid", "file", "clip"},
    "start_frame": {"start_frame", "startframe", "from_frame", "begin_frame"},
    "end_frame": {"end_frame", "endframe", "stop_frame", "stopframe",
                  "to_frame", "finish_frame"},
    "start_time_s": {"start_time_s", "start_s", "start_time", "from_s"},
    "end_time_s": {"end_time_s", "end_s", "stop_time_s", "stop_s", "to_s"},
    "label": {"label", "behavior", "behaviour", "class", "type"},
    "should_use": {"shoulduse", "should_use"},
    "is_circling": {"circling", "is_circling"},
    "verified": {"verified"},
}
_TRUE_STRINGS = {"true", "1", "yes", "y", "t"}


def _truthy(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower().isin(_TRUE_STRINGS)

META_FILENAME = "_conversion_meta.json"
SLOTS_SUFFIX = "_slots.npz"           
PAIR_FEATURE_SUFFIX = "_pairfeat.pickle"  
FEATURE_FILES = ("pairemb",)


def feature_file_suffix(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    if name not in FEATURE_FILES:
        raise ValueError(f"unknown feature file {name!r}; known: {FEATURE_FILES}")
    return f"_{name}.pickle"



def ensure_dlc2action_importable():
    try:
        import matplotlib
        import matplotlib.cm as _cm
        if not hasattr(_cm, "get_cmap"):
            _cm.get_cmap = matplotlib.colormaps.get_cmap
    except ImportError:
        pass
    import dlc2action  # noqa: F401
    return dlc2action


def dlc2action_version() -> Optional[str]:
    try:
        from importlib.metadata import version
        return version("dlc2action")
    except Exception:
        return None


def install_dlc2action_patches() -> bool:
    """Register a csv annotation store that honours ``interactive``.

    Must be called in the process that builds the dataset (train, evaluate,
    predict) BEFORE the Project/Task creates it. Idempotent. Returns True if
    the patch is active.
    """
    ensure_dlc2action_importable()
    from dlc2action import options
    from dlc2action.data.annotation_store import CSVActionSegStore

    current = options.annotation_stores.get("csv")
    if not getattr(current, "_circling_patched", False):
        _install_csv_store_patch(options, CSVActionSegStore)
    _install_sequential_loading_patch()
    _install_ssl_eval_patch()
    logger.debug("installed DLC2Action patches (csv store, sequential "
                 "loading, SSL-free evaluation)")
    return True


def _install_csv_store_patch(options, CSVActionSegStore) -> None:
    class InteractiveCSVActionSegStore(CSVActionSegStore):
        """CSV store that forwards ``interactive`` (DLC2Action 1.0 drops it).

        ``interactive`` must be a named argument so the task dispatcher
        fills it from ``general.interactive``. ``use_hard_negatives`` is
        deliberately NOT forwarded: with exclusive labels DLC2Action would
        write hard negatives as positives.
        """
        _circling_patched = True

        def __init__(self, video_order=None, min_frames=None, max_frames=None,
                     visibility=None, exclusive=True, len_segment=128,
                     overlap=0, behaviors=None, ignored_classes=None,
                     annotation_suffix=None, annotation_path=None,
                     behavior_file=None, correction=None, frame_limit=0,
                     filter_annotated=False, filter_background=False,
                     error_class=None, min_frames_action=None,
                     key_objects=None, visibility_min_score=0.2,
                     visibility_min_frac=0.7, mask=None,
                     default_agent_name="ind0", separator=",", fps=30,
                     interactive=False, *args, **kwargs):
            self._forced_interactive = bool(interactive)
            super().__init__(
                video_order=video_order, min_frames=min_frames,
                max_frames=max_frames, visibility=visibility,
                exclusive=exclusive, len_segment=len_segment, overlap=overlap,
                behaviors=behaviors, ignored_classes=ignored_classes,
                annotation_suffix=annotation_suffix,
                annotation_path=annotation_path, behavior_file=behavior_file,
                correction=correction, frame_limit=frame_limit,
                filter_annotated=filter_annotated,
                filter_background=filter_background, error_class=error_class,
                min_frames_action=min_frames_action, key_objects=key_objects,
                visibility_min_score=visibility_min_score,
                visibility_min_frac=visibility_min_frac, mask=mask,
                default_agent_name=default_agent_name, separator=separator,
                fps=fps, *args, **kwargs)

        def _load_data(self):
            self.interactive = getattr(self, "_forced_interactive", False)
            return super()._load_data()

    options.annotation_stores["csv"] = InteractiveCSVActionSegStore


def _install_sequential_loading_patch() -> None:
    import dlc2action.data.input_store as _input_store
    if not getattr(_input_store.p_map, "_circling_sequential", False):
        def _sequential_map(fn, items, num_cpus=None, **kwargs):
            return [fn(x) for x in items]
        _sequential_map._circling_sequential = True
        _input_store.p_map = _sequential_map


def _install_ssl_eval_patch() -> None:
    from dlc2action.model.base_model import Model
    if getattr(Model.forward, "_circling_ssl_safe", False):
        return
    original = Model.forward

    def forward(self, x, ssl_xs, tag=None):
        missing = (ssl_xs is None or len(ssl_xs) == 0
                   or all(s is None for s in ssl_xs))
        if getattr(self, "ssl_active", False) and missing:
            self.ssl_active = False
            try:
                return original(self, x, ssl_xs, tag=tag)
            finally:
                self.ssl_active = True
        return original(self, x, ssl_xs, tag=tag)

    forward._circling_ssl_safe = True
    Model.forward = forward


def detect_bodyparts(columns: Iterable[str]) -> List[str]:
    cols = set(columns)
    out = []
    for c in columns:
        if c.endswith("_x"):
            bp = c[:-2]
            if f"{bp}_y" in cols and not bp.startswith(("midline_", "qc_")):
                out.append(bp)
    return out


def check_keypoint_parquet_schema(columns: Sequence[str],
                                  path: Union[str, Path] = "") -> None:
    cols = set(columns)
    if {"TrackID_1", "TrackID_2"} & cols or "pair_episode_id" in cols:
        raise ValueError(
            f"{path} looks like a PAIR parquet (TrackID_1/TrackID_2, "
            "pair_episode_id). DLC2Action needs raw keypoints: pass the "
            "single-fish (per-track) parquet instead. Pair kinematics are "
            "recomputed by DLC2Action in interactive mode.")
    need = {"FrameNum", "TrackID"}
    if not need <= cols:
        raise ValueError(f"{path} is missing columns {sorted(need - cols)}")
    if not detect_bodyparts(columns):
        raise ValueError(f"{path} has no <bodypart>_x/<bodypart>_y columns")


def export_dlc_track_h5(
    parquet_path: Union[str, Path],
    out_h5: Union[str, Path],
    scorer: str = "YOLO",
    video_name: Optional[str] = None,
    bodyparts: Optional[Sequence[str]] = None,
    max_individuals: int = N_INDIVIDUALS,
    slot_assignment: str = "tracklet",
    max_gap: int = 15,
    rank_by: str = "X_center",
    frame_offset: int = 0,
    drop_zero_coords: bool = True,
    stride: int = 1,
    float32: bool = True,
    compress: bool = True,
    male_values: Sequence[str] = ("m", "male"),
    switch_factor: float = 1.5,
) -> Dict:
    if slot_assignment not in ("male-partner", "tracklet", "rank"):
        raise ValueError(f"slot_assignment must be 'male-partner', "
                         f"'tracklet' or 'rank', got {slot_assignment!r}")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    parquet_path = Path(parquet_path)
    out_h5 = Path(out_h5)
    video_name = video_name or parquet_path.stem.replace("_keypoints", "")

    logger.info(f"[{video_name}] reading {parquet_path}")
    df = pd.read_parquet(parquet_path)
    if df.empty:
        raise ValueError(f"{parquet_path} contains no rows")
    check_keypoint_parquet_schema(df.columns, parquet_path)

    if bodyparts is None:
        bodyparts = DEFAULT_BODYPARTS
    bodyparts = list(bodyparts)
    missing_bp = [bp for bp in bodyparts
                  if f"{bp}_x" not in df.columns or f"{bp}_y" not in df.columns]
    if missing_bp:
        raise ValueError(
            f"bodyparts missing from {parquet_path.name}: {missing_bp} "
            f"(found: {detect_bodyparts(df.columns)}). Pass bodyparts= "
            "explicitly or fix the export.")

    if frame_offset:
        df["FrameNum"] = df["FrameNum"].astype(np.int64) + frame_offset

    id_col = "TrackUID" if "TrackUID" in df.columns else "TrackID"
    dup = df.duplicated(["FrameNum", id_col])
    if dup.any():
        days = (sorted(df["day_index"].unique().tolist())
                if "day_index" in df.columns else "n/a")
        raise ValueError(
            f"{parquet_path.name}: {int(dup.sum())} duplicate (FrameNum, "
            f"{id_col}) rows. If the parquet spans several recording days "
            f"(day_index values: {days}) FrameNum probably restarts per day; "
            "export one parquet per continuous video instead.")

    tracks_per_frame = df.groupby("FrameNum")[id_col].nunique()
    max_simultaneous = int(tracks_per_frame.max())
    n_slots = int(max_individuals)
    individuals = [f"ind{i}" for i in range(n_slots)]

    frame_min = int(df["FrameNum"].min())
    frame_max = int(df["FrameNum"].max())
    n_frames_full = frame_max - frame_min + 1
    fps_full = _infer_fps(df)
    df["_frame"] = df["FrameNum"].astype(np.int64) - frame_min
    if stride > 1:
        df = df[df["_frame"] % stride == 0].copy()
        df["_frame"] = df["_frame"] // stride
    n_frames = (n_frames_full - 1) // stride + 1
    fps = (fps_full / stride) if fps_full else None

    columns = pd.MultiIndex.from_product(
        [[scorer], individuals, bodyparts, ["x", "y", "likelihood"]],
        names=["scorer", "individuals", "bodyparts", "coords"])
    dtype = np.float32 if float32 else np.float64

    slot_stats = {}
    if slot_assignment == "male-partner":
        if n_slots != 2:
            raise ValueError("male-partner assignment needs max_individuals=2")
        slots, slot_stats = _assign_male_partner_slots(
            df, id_col=id_col, male_values=male_values,
            switch_factor=switch_factor, video=video_name)
        n_dropped, n_tracks = 0, int(df[id_col].nunique())
    elif slot_assignment == "tracklet":
        slots, n_dropped, n_tracks = _assign_tracklet_slots(
            df, n_slots, id_col=id_col, max_gap=max_gap)
    else:
        key = rank_by if rank_by in df.columns else id_col
        df = df.sort_values(["_frame", key], kind="mergesort")
        slots = df.groupby("_frame", sort=False).cumcount().to_numpy()
        n_dropped = int((slots >= n_slots).sum())
        slots = np.where(slots >= n_slots, -1, slots)
        n_tracks = int(df[id_col].nunique())
    df["_slot"] = slots
    df = df[df["_slot"] >= 0]
    slot_uid = np.full((n_frames, n_slots), -1, dtype=np.int64)
    slot_uid[df["_frame"].to_numpy(np.int64), df["_slot"].to_numpy(np.int64)] = \
        df[id_col].to_numpy(np.int64)
    slots_path = out_h5.with_name(f"{video_name}{SLOTS_SUFFIX}")
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(slots_path, uid=slot_uid, id_col=np.array(id_col))

    arr = np.full((n_frames, n_slots, len(bodyparts), 3), np.nan, dtype=dtype)
    arr[..., 2] = 0.0
    lin = (df["_frame"].to_numpy(np.int64) * n_slots
           + df["_slot"].to_numpy(np.int64))
    flat = arr.reshape(-1, len(bodyparts), 3)
    for i, bp in enumerate(bodyparts):
        x = pd.to_numeric(df[f"{bp}_x"], errors="coerce").to_numpy(dtype)
        y = pd.to_numeric(df[f"{bp}_y"], errors="coerce").to_numpy(dtype)
        miss = np.isnan(x) | np.isnan(y)
        if drop_zero_coords:
            miss |= (x <= 0.0) & (y <= 0.0)          
        flat[lin, i, 0] = np.where(miss, np.nan, x)
        flat[lin, i, 1] = np.where(miss, np.nan, y)
        flat[lin, i, 2] = (~miss).astype(dtype)

    present = arr[..., 2].max(axis=2) > 0          
    frac_both = float(present.all(axis=1).mean())
    frac_any = float(present.any(axis=1).mean())
    x_max = float(np.nanmax(arr[..., 0])) if np.isfinite(arr[..., 0]).any() else 0.0
    y_max = float(np.nanmax(arr[..., 1])) if np.isfinite(arr[..., 1]).any() else 0.0

    out = pd.DataFrame(arr.reshape(n_frames, -1), index=range(n_frames),
                       columns=columns)
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    kw = {"complib": "blosc", "complevel": 5} if compress else {}
    out.to_hdf(out_h5, key="df_with_missing", mode="w", **kw)
    size_mb = out_h5.stat().st_size / 1e6

    logger.info(
        f"[{video_name}] {n_frames} rows (frames {frame_min}-{frame_max}, "
        f"stride {stride}), {n_tracks} tracks, max {max_simultaneous} "
        f"simultaneous -> {n_slots} slots; both fish visible in "
        f"{100 * frac_both:.1f}% of rows; {size_mb:.0f} MB -> {out_h5}")
    if n_dropped:
        logger.warning(
            f"[{video_name}] dropped {n_dropped} detection(s) when more than "
            f"{n_slots} tracks were present (spurious tracks?)")
    if fps_full is None:
        logger.warning(f"[{video_name}] could not infer fps (no Time_s); "
                       "pass --fps to export-events/score later")

    return {"video": video_name, "n_frames": int(n_frames),
            "n_slots": n_slots, "individuals": individuals,
            "bodyparts": bodyparts, "fps": fps, "fps_full": fps_full,
            "stride": int(stride), "frame_min": frame_min,
            "frame_max": frame_max, "n_frames_original": int(n_frames_full),
            "slot_assignment": slot_assignment, "max_gap": int(max_gap),
            "slot_stats": slot_stats, "slots_file": str(slots_path),
            "id_col": id_col,
            "n_tracks": n_tracks, "max_simultaneous_tracks": max_simultaneous,
            "n_dropped_detections": int(n_dropped),
            "frac_rows_both_fish": round(frac_both, 4),
            "frac_rows_any_fish": round(frac_any, 4),
            "x_max": x_max, "y_max": y_max,
            "parquet": str(parquet_path), "h5": str(out_h5)}


def _assign_male_partner_slots(df: pd.DataFrame, id_col: str = "TrackUID",
                               male_values: Sequence[str] = ("m", "male"),
                               switch_factor: float = 1.5, video: str = ""):
    if "Sex" not in df.columns:
        raise ValueError("male-partner assignment needs a 'Sex' column")
    for c in ("X_center", "Y_center"):
        if c not in df.columns:
            raise ValueError(f"male-partner assignment needs '{c}'")
    male_values = {str(v).strip().lower() for v in male_values}
    sex = df["Sex"].astype(str).str.strip().str.lower()
    row_male = sex.isin(male_values)
    if not row_male.any():
        raise ValueError(
            f"[{video}] no detections with Sex in {sorted(male_values)} "
            f"(values found: {sorted(sex.unique())[:10]}); pass --male-values")
    # sex per TRACK by majority vote (robust to per-frame misclassification)
    track_male = row_male.groupby(df[id_col].to_numpy()).mean() > 0.5
    is_male = df[id_col].map(track_male).fillna(False).to_numpy(bool)

    order = np.argsort(df["_frame"].to_numpy(), kind="mergesort")
    frames = df["_frame"].to_numpy()[order]
    uids = df[id_col].to_numpy()[order]
    male = is_male[order]
    xs = df["X_center"].to_numpy(float)[order]
    ys = df["Y_center"].to_numpy(float)[order]
    slots_sorted = np.full(len(df), -1, dtype=np.int64)

    male_uid = partner_uid = None
    male_xy = None
    n_frames_male = n_frames_partner = n_switches = n_multi_male = 0
    i, n = 0, len(frames)
    while i < n:
        j = i
        f = frames[i]
        while j < n and frames[j] == f:
            j += 1
        ks = np.arange(i, j)
        mk = ks[male[i:j]]
        fk = ks[~male[i:j]]
        m = None
        if len(mk):
            if len(mk) > 1:
                n_multi_male += 1
            same = mk[uids[mk] == male_uid] if male_uid is not None else mk[:0]
            if len(same):
                m = same[0]
            elif male_xy is not None:
                d = np.hypot(xs[mk] - male_xy[0], ys[mk] - male_xy[1])
                m = mk[np.argmin(d)]
            else:
                m = mk[0]
            male_uid = uids[m]
            male_xy = (xs[m], ys[m])
            slots_sorted[m] = 0
            n_frames_male += 1
        p = None
        if len(fk):
            cur = fk[uids[fk] == partner_uid] if partner_uid is not None else fk[:0]
            if m is not None:
                d = np.hypot(xs[fk] - xs[m], ys[fk] - ys[m])
                nearest = int(np.argmin(d))
                if len(cur):
                    dcur = d[np.where(fk == cur[0])[0][0]]
                    p = cur[0] if dcur <= switch_factor * d[nearest] \
                        else fk[nearest]
                else:
                    p = fk[nearest]
            elif len(cur):
                p = cur[0]          
        if p is not None:
            if partner_uid is not None and uids[p] != partner_uid:
                n_switches += 1
            partner_uid = uids[p]
            slots_sorted[p] = 1
            n_frames_partner += 1
        i = j
    slots = np.empty_like(slots_sorted)
    slots[order] = slots_sorted
    n_unique = len(np.unique(frames))
    stats = {"frac_frames_male": round(n_frames_male / max(1, n_unique), 4),
             "frac_frames_partner": round(n_frames_partner / max(1, n_unique), 4),
             "partner_switches": int(n_switches),
             "frames_multiple_male_tracks": int(n_multi_male),
             "n_male_tracks": int(track_male.sum()),
             "n_female_tracks": int((~track_male).sum())}
    logger.info(f"[{video}] male-partner slots: male in "
                f"{100 * stats['frac_frames_male']:.1f}% of detected frames, "
                f"{stats['n_male_tracks']} male / {stats['n_female_tracks']} "
                f"female track(s), {n_switches} partner switches, "
                f"{n_multi_male} frames with >1 male track")
    if n_multi_male > 0.05 * max(1, n_unique):
        logger.warning(f"[{video}] >1 male-classified track in "
                       f"{n_multi_male} frames — check the sex classifier")
    return slots, stats


def _assign_tracklet_slots(df: pd.DataFrame, n_slots: int,
                           id_col: str = "TrackUID", max_gap: int = 15):
    order = np.argsort(df["_frame"].to_numpy(), kind="mergesort")
    frames = df["_frame"].to_numpy()[order]
    ids = df[id_col].to_numpy()[order]
    slots = np.full(len(df), -1, dtype=np.int64)
    track_slot: Dict = {}
    last_seen: Dict = {}
    free: deque = deque(range(n_slots))
    n_dropped = 0
    i, n = 0, len(frames)
    while i < n:
        j = i
        f = frames[i]
        while j < n and frames[j] == f:
            j += 1
        for t in [t for t in track_slot if f - last_seen[t] > max_gap]:
            free.append(track_slot.pop(t))
            del last_seen[t]
        for k in range(i, j):
            t = ids[k]
            if t in track_slot:
                slots[order[k]] = track_slot[t]
                last_seen[t] = f
            elif free:
                s = free.popleft()
                track_slot[t] = s
                last_seen[t] = f
                slots[order[k]] = s
            else:
                n_dropped += 1
        i = j
    return slots, n_dropped, len(set(ids.tolist()))


def _infer_fps(df: pd.DataFrame) -> Optional[float]:
    if "Time_s" not in df.columns:
        return None
    t = (df[["FrameNum", "Time_s"]].dropna().drop_duplicates("FrameNum")
         .sort_values("FrameNum"))
    if len(t) < 2:
        return None
    df_ = np.diff(t["FrameNum"].to_numpy(dtype=float))
    dt = np.diff(t["Time_s"].to_numpy(dtype=float))
    ok = (df_ > 0) & (dt > 0)
    if not ok.any():
        return None
    return float(1.0 / np.median(dt[ok] / df_[ok]))



PAIR_SOURCE_COLUMNS = [
    "FrameNum", "TrackUID_1", "TrackUID_2", "centroid_distance_px",
    "dist_nose1_nose2_px", "dist_nose1_tailtip2_px", "dist_nose2_tailtip1_px",
    "dist_nose1_spine4of2_px", "dist_nose2_spine4of1_px",
    "rel_heading_rad", "bearing_1_rad", "bearing_2_rad",
    "d_heading_1_rad_s", "d_heading_2_rad_s", "d_orbital_rad_s",
]


def _pair_feature_table(chunk: pd.DataFrame, male_first: np.ndarray):
    def pick(a, b):
        x, y = chunk[a].to_numpy(float), chunk[b].to_numpy(float)
        return np.where(male_first, x, y), np.where(male_first, y, x)

    f = {}
    f["centroid_dist_px"] = chunk["centroid_distance_px"].to_numpy(float)
    f["nose_nose_px"] = chunk["dist_nose1_nose2_px"].to_numpy(float)
    f["noseM_tailF_px"], f["noseF_tailM_px"] = pick(
        "dist_nose1_tailtip2_px", "dist_nose2_tailtip1_px")
    f["min_head_to_tail_px"] = np.fmin(f["noseM_tailF_px"], f["noseF_tailM_px"])
    f["noseM_spine4F_px"], f["noseF_spine4M_px"] = pick(
        "dist_nose1_spine4of2_px", "dist_nose2_spine4of1_px")
    rel = chunk["rel_heading_rad"].to_numpy(float)
    f["cos_rel_heading"], f["abs_sin_rel_heading"] = np.cos(rel), np.abs(np.sin(rel))
    bm, bf = pick("bearing_1_rad", "bearing_2_rad")
    f["cos_bearing_M"], f["abs_sin_bearing_M"] = np.cos(bm), np.abs(np.sin(bm))
    f["cos_bearing_F"], f["abs_sin_bearing_F"] = np.cos(bf), np.abs(np.sin(bf))
    hm, hf = pick("d_heading_1_rad_s", "d_heading_2_rad_s")
    f["abs_turn_rate_M"], f["abs_turn_rate_F"] = np.abs(hm), np.abs(hf)
    f["abs_orbit_rate"] = np.abs(chunk["d_orbital_rad_s"].to_numpy(float))
    return f


def export_pair_features(
    pairs_parquet: Union[str, Path],
    meta_entry: Dict,
    out_path: Union[str, Path],
    batch_rows: int = 2_000_000,
) -> Dict:
    import pyarrow.parquet as pq
    pairs_parquet, out_path = Path(pairs_parquet), Path(out_path)
    video = meta_entry["video"]
    slots_file = meta_entry.get("slots_file")
    slots_file = Path(slots_file) if slots_file else None
    if slots_file is None or not slots_file.is_file():
        raise FileNotFoundError(
            f"[{video}] slot file not found: re-run "
            "convert-data with the same arguments (it now also saves which "
            "track sits in each slot). The H5, annotations, splits and "
            "trained folds stay valid: the conversion is deterministic.")
    uid = np.load(slots_file)["uid"]
    n = int(meta_entry["n_frames"])
    if uid.shape[0] != n:
        raise ValueError(f"[{video}] slot file has {uid.shape[0]} rows, H5 "
                         f"has {n}: re-run convert-data")
    stride = int(meta_entry.get("stride", 1))
    fmin = int(meta_entry.get("frame_min", 0))

    pf = pq.ParquetFile(pairs_parquet)
    have = set(pf.schema_arrow.names)
    if {"Nose_x", "TrackID"} <= have and "TrackID_1" not in have:
        raise ValueError(f"{pairs_parquet.name} is a single-fish parquet; "
                         "convert-pairs needs the PAIR parquet")
    missing = [c for c in PAIR_SOURCE_COLUMNS if c not in have]
    if missing:
        raise ValueError(f"{pairs_parquet.name} lacks columns {missing}")

    feats, rows_hit, n_seen = None, None, 0
    for batch in pf.iter_batches(batch_size=batch_rows,
                                 columns=PAIR_SOURCE_COLUMNS):
        chunk = batch.to_pandas()
        n_seen += len(chunk)
        off = chunk["FrameNum"].to_numpy(np.int64) - fmin
        r = off // stride
        ok = (off >= 0) & (off % stride == 0) & (r < n)
        chunk, r = chunk[ok], r[ok]
        if chunk.empty:
            continue
        a = chunk["TrackUID_1"].to_numpy(np.int64)
        b = chunk["TrackUID_2"].to_numpy(np.int64)
        m, p = uid[r, 0], uid[r, 1]
        direct = (a == m) & (b == p) & (m >= 0) & (p >= 0)
        swapped = (a == p) & (b == m) & (m >= 0) & (p >= 0)
        keep = direct | swapped
        if not keep.any():
            continue
        chunk, r, direct = chunk[keep], r[keep], direct[keep]
        table = _pair_feature_table(chunk, direct)
        if feats is None:
            names = list(table)
            feats = np.full((n, len(names)), np.nan, dtype=np.float64)
            rows_hit = np.zeros(n, dtype=bool)
        # first matching row per H5 row wins
        first = ~rows_hit[r]
        _, uniq = np.unique(r, return_index=True)
        sel = np.zeros(len(r), dtype=bool)
        sel[uniq] = True
        sel &= first
        feats[r[sel]] = np.column_stack([table[k] for k in names])[sel]
        rows_hit[r[sel]] = True

    if feats is None:
        raise ValueError(
            f"[{video}] no pair row matches the male + partner in any frame. "
            "Check that the pair parquet belongs to this video and uses the "
            "same TrackUIDs as the single-fish parquet.")
    present = rows_hit.astype(np.float64)
    for j in range(feats.shape[1]):
        col = feats[:, j]
        good = np.isfinite(col)
        col[~good] = np.median(col[good]) if good.any() else 0.0
    out = np.column_stack([feats, present]).astype(np.float32)
    names = names + ["pair_present"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        pickle.dump({PAIR_CLIP: out}, fh)
    both = (uid[:, 0] >= 0) & (uid[:, 1] >= 0)
    stats = {"pair_features": names, "n_pair_rows_read": int(n_seen),
             "frac_rows_with_pair": round(float(rows_hit.mean()), 4),
             "frac_slot_rows_with_pair": round(
                 float(rows_hit[both].mean()) if both.any() else 0.0, 4),
             "pair_parquet": str(pairs_parquet), "pair_file": str(out_path)}
    logger.info(f"[{video}] pair features {out.shape} -> {out_path.name}; "
                f"pair found in {100 * stats['frac_rows_with_pair']:.1f}% of "
                f"rows ({100 * stats['frac_slot_rows_with_pair']:.1f}% of rows "
                "where both slots are filled)")
    return stats


# ---------------------------------------------------------------------------
# 2. Names + ground truth
# ---------------------------------------------------------------------------

def normalize_ground_truth_columns(gt: pd.DataFrame) -> pd.DataFrame:
    """Rename ground-truth columns to the canonical schema."""
    lookup = {}
    for orig in gt.columns:
        low = orig.strip().lower()
        for canon, aliases in _GT_ALIASES.items():
            if low in aliases:
                lookup[orig] = canon
    return gt.rename(columns=lookup)


def read_ground_truth(gt_csv: Union[str, Path]) -> pd.DataFrame:
    """Read + validate the ground-truth CSV (frame- or time-based)."""
    gt = pd.read_csv(gt_csv, dtype=str)
    gt = normalize_ground_truth_columns(gt)
    if "video" not in gt.columns:
        raise ValueError("ground truth needs a Video column; got "
                         f"{list(gt.columns)}")
    gt = gt.dropna(how="all")
    gt["video"] = gt["video"].str.strip()
    # Team ground truth (merged_spans.csv): ShouldUse=false rows are
    # superseded duplicates and Circling=false rows were reviewed as not
    # circling -> both dropped (background). Verified (dlc2action-branch
    # format): true = circling; false/blank = not confirmed -> "unknown",
    # which DLC2Action ignores in the loss, as the team's label policy does.
    n0 = len(gt)
    if "should_use" in gt.columns:
        gt = gt[_truthy(gt["should_use"])]
    if "is_circling" in gt.columns:
        gt = gt[_truthy(gt["is_circling"])]
    if len(gt) < n0:
        logger.info(f"ground truth: kept {len(gt)} of {n0} rows "
                    "(ShouldUse/Circling flags)")
    if "verified" in gt.columns:
        unverified = ~_truthy(gt["verified"])
        if unverified.any():
            gt = gt.copy()
            if "label" not in gt.columns:
                gt["label"] = ""
            gt.loc[unverified, "label"] = ERROR_CLASS
            logger.info(f"ground truth: {int(unverified.sum())} unverified "
                        f"rows -> '{ERROR_CLASS}' (ignored in the loss)")
    gt = gt.reset_index(drop=True)
    frame_based = {"start_frame", "end_frame"} <= set(gt.columns)
    time_based = {"start_time_s", "end_time_s"} <= set(gt.columns)
    if not (frame_based or time_based):
        raise ValueError("ground truth needs StartFrame/StopFrame or "
                         f"StartTime_s/StopTime_s; got {list(gt.columns)}")
    if frame_based:
        for c in ("start_frame", "end_frame"):
            gt[c] = pd.to_numeric(gt[c].str.strip(), errors="raise")
    gt.attrs["frame_based"] = frame_based
    return gt


def resolve_video_names(stems: Sequence[str], gt_videos: Sequence[str],
                        name_map: Optional[Dict[str, str]] = None
                        ) -> Dict[str, str]:
    """Map parquet stems to ground-truth video ids.

    Order: explicit name_map -> exact match -> unique GT id that shares the
    leading numeric code (e.g. '0028_keypoints' -> '0028_vid') or is a
    substring of the stem. Raises if any stem stays ambiguous/unmatched while
    ground truth is available.
    """
    name_map = name_map or {}
    gt_videos = list(dict.fromkeys(gt_videos))
    out = {}
    for stem in stems:
        base = stem.replace("_keypoints", "")
        if stem in name_map or base in name_map:
            out[stem] = name_map.get(stem, name_map.get(base))
            continue
        if not gt_videos:
            out[stem] = base
            continue
        if base in gt_videos:
            out[stem] = base
            continue
        code = re.match(r"\d+", base)
        cands = [v for v in gt_videos
                 if (code and re.match(r"\d+", v)
                     and re.match(r"\d+", v).group() == code.group())
                 or v in stem]
        if len(cands) == 1:
            out[stem] = cands[0]
        else:
            raise ValueError(
                f"cannot match parquet '{stem}' to a ground-truth video "
                f"(candidates {cands or 'none'} of {gt_videos}); pass "
                f"--name-map {base}=<video id>")
    targets = list(out.values())
    if len(set(targets)) != len(targets):
        raise ValueError(f"two parquets map to the same video: {out}")
    return out


def ground_truth_to_annotations(
    gt_csv: Union[str, Path],
    out_dir: Union[str, Path],
    meta: Dict[str, Dict],
    default_fps: Optional[float] = None,
    default_label: str = "circling",
    merge_existing: bool = False,
) -> List[Path]:
    """Convert the ground-truth CSV into DLC2Action csv annotation files.

    Frame-based GT is on the ORIGINAL video axis; it is shifted by the
    per-video ``frame_min`` and divided by ``stride`` (from ``meta``) so it
    indexes H5 rows. Time-based GT uses the effective fps.

    Labels: missing -> ``default_label``; ``negative ...`` kept (explicit
    background); anything starting with ``unknown`` -> ``unknown`` (the
    configured error_class, ignored in the loss).

    Writes ``{video}.csv`` (behavior,start,end; end exclusive) for every
    video in ``meta`` (an empty file for a video without GT intervals is
    valid: the whole video is background). Videos in GT without an H5 raise.
    """
    gt = read_ground_truth(gt_csv)
    frame_based = gt.attrs["frame_based"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = sorted(set(gt["video"]) - set(meta))
    if missing:
        raise ValueError(
            f"ground-truth video(s) {missing} have no converted H5 (known: "
            f"{sorted(meta)}). DLC2Action joins data and annotations by exact "
            "file stem; fix with convert-data --name-map.")

    written = []
    for video in sorted(meta):
        m = meta[video]
        grp = gt[gt["video"] == video].copy()
        stride = int(m.get("stride", 1))
        fmin = int(m.get("frame_min", 0))
        n_max = int(m["n_frames"])
        rows = []
        for _, r in grp.iterrows():
            if frame_based:
                s = int((float(r["start_frame"]) - fmin) // stride)
                # StopFrame is inclusive on the video axis -> exclusive end
                e = int(np.ceil((float(r["end_frame"]) - fmin + 1) / stride))
            else:
                fps = m.get("fps") or default_fps
                if not fps:
                    raise ValueError(f"fps unknown for '{video}'")
                s = int(float(r["start_time_s"]) * fps)
                e = int(np.ceil(float(r["end_time_s"]) * fps))
            label = default_label
            if "label" in grp.columns and isinstance(r["label"], str) \
                    and r["label"].strip():
                label = r["label"].strip()
            if label.lower().startswith("unknown"):
                label = ERROR_CLASS
            if e <= s:
                logger.warning(f"{video}: dropped empty interval [{s},{e})")
                continue
            if s < 0 or s >= n_max:
                raise ValueError(
                    f"{video}: interval starting at H5 row {s} is outside the "
                    f"H5 (0..{n_max - 1}). Original-axis frames "
                    f"{m.get('frame_min')}..{m.get('frame_max')} were "
                    "converted; check that the parquet covers the GT frames "
                    "and that FrameNum matches the video frame numbering.")
            if e > n_max:
                logger.warning(f"{video}: interval [{s},{e}) clipped to "
                               f"{n_max}")
                e = n_max
            rows.append({ANN_COL_BEHAVIOR: label, ANN_COL_START: s,
                         ANN_COL_END: e})

        path = out_dir / f"{video}.csv"
        if merge_existing and path.exists():
            old = pd.read_csv(path)
            old.columns = old.columns.str.strip().str.lower()
            if {ANN_COL_BEHAVIOR, ANN_COL_START, ANN_COL_END} <= set(old.columns):
                rows = (old[[ANN_COL_BEHAVIOR, ANN_COL_START, ANN_COL_END]]
                        .to_dict("records") + rows)
        out = _clean_intervals(pd.DataFrame(
            rows, columns=[ANN_COL_BEHAVIOR, ANN_COL_START, ANN_COL_END]),
            video=video)
        out.to_csv(path, index=False)
        written.append(path)
        n_pos = int((~out[ANN_COL_BEHAVIOR].str.startswith(
            ("negative", ERROR_CLASS))).sum()) if len(out) else 0
        logger.info(f"{video}: {len(out)} interval(s), {n_pos} positive "
                    f"-> {path}")
    return written


def _clean_intervals(df: pd.DataFrame, video: str = "") -> pd.DataFrame:
    """Sort, merge overlapping same-label intervals, warn on cross-label."""
    if df.empty:
        return df
    df = df.sort_values([ANN_COL_START, ANN_COL_END]).reset_index(drop=True)
    merged = []
    for _, r in df.iterrows():
        s, e = int(r[ANN_COL_START]), int(r[ANN_COL_END])
        lab = r[ANN_COL_BEHAVIOR]
        if merged and lab == merged[-1][ANN_COL_BEHAVIOR] \
                and s <= merged[-1][ANN_COL_END]:
            merged[-1][ANN_COL_END] = max(merged[-1][ANN_COL_END], e)
            continue
        if merged and s < merged[-1][ANN_COL_END]:
            logger.warning(f"{video}: overlapping intervals with different "
                           f"labels at row {s} — please review")
        merged.append({ANN_COL_BEHAVIOR: lab, ANN_COL_START: s,
                       ANN_COL_END: e})
    return pd.DataFrame(merged)


def load_meta(data_dir: Union[str, Path]) -> Dict[str, Dict]:
    p = Path(data_dir) / META_FILENAME
    if not p.exists():
        raise FileNotFoundError(f"{p} not found — run python -m models.mstcn_dlc2action.train "
                                "(it converts the data first)")
    return json.loads(p.read_text(encoding="utf-8"))


def read_positive_intervals(ann_dir: Union[str, Path],
                            videos: Sequence[str],
                            behavior: str = "circling"
                            ) -> Dict[str, List[Tuple[int, int]]]:
    """Positive [start, end) intervals per video from the csv annotations."""
    out = {}
    for v in videos:
        p = Path(ann_dir) / f"{v}.csv"
        if not p.exists():
            out[v] = []
            continue
        d = pd.read_csv(p)
        d.columns = d.columns.str.strip().str.lower()
        d = d[d[ANN_COL_BEHAVIOR] == behavior]
        out[v] = sorted((int(s), int(e)) for s, e in
                        zip(d[ANN_COL_START], d[ANN_COL_END]))
    return out


def read_ignored_intervals(ann_dir, videos):
    out = {}
    for v in videos:
        p = Path(ann_dir) / f"{v}.csv"
        if not p.exists():
            out[v] = []
            continue
        d = pd.read_csv(p)
        d.columns = d.columns.str.strip().str.lower()
        d = d[d[ANN_COL_BEHAVIOR] == ERROR_CLASS]
        out[v] = [(int(s), int(e)) for s, e in
                  zip(d[ANN_COL_START], d[ANN_COL_END])]
    return out


# ---------------------------------------------------------------------------
# 3. Project config
# ---------------------------------------------------------------------------

def default_project_parameters(
    data_dir: Union[str, Path],
    behaviors: Optional[Sequence[str]] = None,
    len_segment: int = 128,
    overlap: int = 64,
    min_frames_action: int = 5,
    canvas_shape: Optional[Sequence[int]] = None,
    partition_method: str = "file",
    split_path: Optional[Union[str, Path]] = None,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    num_epochs: int = 100,
    batch_size: int = 64,
    device: str = "auto",
    num_cpus: int = 8,
    likelihood_threshold: float = 0.5,
    ignored_bodyparts: Optional[Sequence[str]] = None,
    model_save_epochs: int = 10,
    pair_features: bool = False,
    experiment: Optional[Dict] = None,
    feature_file: Optional[str] = None,
) -> Dict:
    """Parameter update dict for ``Project.update_parameters()``.

    pair_features=True loads ``{video}_pairfeat.pickle`` (convert-pairs)
    through DLC2Action's ``data.feature_suffix`` and appends it to the
    kinematic features of the ind0+ind1 clip; False clears it explicitly.
    feature_file="pairemb" loads ``{video}_pairemb.pickle`` instead (pair
    features + image-embedding PCs in one file); exclusive with pair_features.

    Fixed choices (see module docstring for why):
    exclusive=True, interactive=True, default_agent_name="ind0+ind1",
    use_hard_negatives=False, error_class="unknown", fps=1,
    filter_background=False, H5 and csv in the same ``data_dir``.
    """
    behaviors = sorted(set(behaviors or ["circling"]))
    data_dir = str(Path(data_dir).resolve())
    if pair_features and feature_file:
        raise ValueError("use pair_features OR feature_file, not both: the "
                         "feature file already contains the pair features")
    suffix = (feature_file_suffix(feature_file) if feature_file
              else PAIR_FEATURE_SUFFIX if pair_features else None)
    params = {
        "general": {
            "exclusive": True,
            "feature_extraction": "kinematic",
            "interactive": True,
            "len_segment": int(len_segment),
            "overlap": int(overlap),
            "num_cpus": int(num_cpus),
            "only_load_annotated": True,
        },
        "data": {
            "data_path": data_dir,
            "annotation_path": data_dir,
            "behaviors": set(behaviors),
            "data_suffix": {".h5"},
            "annotation_suffix": {".csv"},
            "canvas_shape": [int(v) for v in (canvas_shape or [1280, 720])],
            "fps": 1,
            "min_frames_action": int(min_frames_action),
            "filter_annotated": True,
            "filter_background": False,
            "use_hard_negatives": False,
            "error_class": ERROR_CLASS,
            "default_agent_name": PAIR_CLIP,
            "likelihood_threshold": float(likelihood_threshold),
            "ignored_bodyparts": (set(ignored_bodyparts)
                                  if ignored_bodyparts else None),
            "feature_suffix": suffix,
        },
        "training": {
            "partition_method": partition_method,
            "val_frac": float(val_frac),
            "test_frac": float(test_frac),
            "num_epochs": int(num_epochs),
            "batch_size": int(batch_size),
            "device": device,
            "model_save_epochs": int(model_save_epochs),
        },
    }
    # DLC2Action macro-averages metrics over ALL classes incl. "other" (0),
    # which the huge background class inflates. Report behaviour classes only.
    params["metrics"] = {m: {"ignored_classes": {0}}
                         for m in ("f1", "precision", "recall")}
    if partition_method == "file":
        if not split_path:
            raise ValueError("partition_method 'file' needs split_path "
                             "(run make-split)")
        params["training"]["split_path"] = str(Path(split_path).resolve())
        params["training"]["val_frac"] = 0.0
        params["training"]["test_frac"] = 0.0
    # Experiment settings (augmentations, SSL, model, feature keys). Always
    # applied, baseline values included: DLC2Action persists every update in
    # the project config, so a preset left out would otherwise carry over
    # from the previous run.
    _deep_merge(params, copy.deepcopy(
        experiment if experiment is not None else experiment_parameters()))
    return params


# ---------------------------------------------------------------------------
# 3b. Experiment presets (checked against the DLC2Action 1.0 source)
# ---------------------------------------------------------------------------
# * MS-TCN3 only has a feature extractor (needed for ANY SSL task) with
#   model.PG_in_FE = true or num_layers_S > 0; the defaults give an identity
#   extractor ("MS-TCN without a feature extractor -> no SSL possible!").
# * augmentations.augmentations is used as given; the default list only
#   applies when that set is empty. Augmentations transform coordinate-type
#   kinematic keys only, so the (rotation/mirror-invariant) pair features
#   loaded via feature_suffix are left untouched, as they should be.
# * Never use the 'switch' augmentation: it swaps ind0/ind1, i.e. the male
#   and partner roles the pair features are ordered by.
# * Loss: focal + inverse class weights are DLC2Action defaults already.

FEATURE_KEYS_DEFAULT = frozenset({
    "coord_diff", "center", "intra_distance", "speed_direction",
    "speed_value", "angle_speeds", "acc_joints", "inter_distance"})

# What every run so far has used (project defaults). Re-applied each run.
BASELINE_EXPERIMENT = {
    "general": {"ssl": None},
    "model": {"PG_in_FE": False, "num_layers_S": 0},
    "augmentations": {
        "augmentations": {"add_noise", "mirror"},
        "use_default_augmentations": True,
        "rotation_limits": [-1.57, 1.57],
        "mirror_dim": {0},
    },
    "training": {"main_task_on": True, "ssl_on": True,
                 "freeze_features": False},
    "features": {"keys": set(FEATURE_KEYS_DEFAULT)},
}

PRESETS = {
    # top-down tank: any rotation and both mirror axes are valid
    "aug": {"augmentations": {
        "augmentations": {"rotate", "mirror", "shift", "add_noise"},
        "rotation_limits": [-math.pi, math.pi],
        "mirror_dim": {0, 1}}},
    # SSL on the prediction-generator features
    # general.ssl must be a LIST: the config comment says "set", but
    # TaskDispatcher._construct_ssl assigns ssl_list[i] (a set -> TypeError).
    # Per-frame heads only: see FIXED_LENGTH_SSL.
    "ssl": {"model": {"PG_in_FE": True},
            "general": {"ssl": ["masked_features", "masked_frames"]}},
    # drop absolute position: tests whether the pose model leans on where in
    # the tank things happen (pose-only failed across videos)
    "nocenter": {"features": {"keys": set(FEATURE_KEYS_DEFAULT - {"center"})}},
}

# SSL heads built for exactly len_segment frames (FeatureExtractorTCN /
# MFeatureExtractorTCN / reshapes by segment length). Training feeds them
# temporal_subsampling_size * len_segment frames (0.85 by default), which
# crashes with "expected input[..., 768, 1] to have 928 channels" at 128.
FIXED_LENGTH_SSL = {"contrastive", "contrastive_masked", "pairwise",
                    "pairwise_masked", "reverse", "order"}

# config entries DLC2Action reads as sets (YAML lists are converted)
SET_PARAMS = {("general", "metric_functions"),
              ("augmentations", "augmentations"),
              ("augmentations", "mirror_dim"), ("features", "keys"),
              ("data", "behaviors"), ("data", "data_suffix"),
              ("data", "annotation_suffix")}


def _deep_merge(dst: Dict, src: Dict) -> Dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def parse_param_override(text: str) -> Tuple[List[str], object]:
    """'model.num_f_maps=64' -> (['model', 'num_f_maps'], 64). YAML values."""
    import yaml
    if "=" not in text:
        raise ValueError(f"--param must be section.key=value, got {text!r}")
    key, raw = text.split("=", 1)
    path = [p for p in key.strip().split(".") if p]
    if len(path) < 2:
        raise ValueError(f"--param key needs a section, e.g. model.num_f_maps; "
                         f"got {key!r}")
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        # YAML 1.1 reads "5e-4" as a string (it wants "5.0e-4")
        try:
            value = int(value) if re.fullmatch(r"[+-]?\d+", value) else float(value)
        except ValueError:
            pass
    if (path[0], path[-1]) in SET_PARAMS and isinstance(value, list):
        value = set(value)
    if tuple(path) == ("general", "ssl") and isinstance(value, str):
        value = [value]                      # one SSL task -> list
    return path, value


def experiment_parameters(presets: Sequence[str] = (),
                          overrides: Sequence[str] = ()) -> Dict:
    """Baseline experiment block + presets (in order) + --param overrides."""
    out = copy.deepcopy(BASELINE_EXPERIMENT)
    for p in presets:
        if p not in PRESETS:
            raise ValueError(f"unknown preset {p!r}; choose from {sorted(PRESETS)}")
        _deep_merge(out, copy.deepcopy(PRESETS[p]))
    for o in overrides:
        path, value = parse_param_override(o)
        if path[-1] == "augmentations" and isinstance(value, set) and "switch" in value:
            raise ValueError("the 'switch' augmentation swaps male and partner; "
                             "not allowed with male-first pair features")
        d = out
        for k in path[:-1]:
            d = d.setdefault(k, {})
        d[path[-1]] = value
    model_name = out.get("general", {}).get("model_name", "ms_tcn3")
    if model_name != "ms_tcn3":
        explicit = {tuple(parse_param_override(o)[0]) for o in overrides}
        for k in ("PG_in_FE", "num_layers_S"):
            if ("model", k) not in explicit:
                out["model"].pop(k, None)
        if not out["model"]:
            out.pop("model")
    fixed = FIXED_LENGTH_SSL & set(out.get("general", {}).get("ssl") or [])
    tss = out.get("training", {}).get("temporal_subsampling_size", 0.85)
    if fixed and tss is not None and float(tss) < 1:
        raise ValueError(
            f"SSL task(s) {sorted(fixed)} need whole segments, but training "
            f"samples {tss} of each segment's frames (DLC2Action builds those "
            "heads for exactly len_segment frames and crashes). Use per-frame "
            "SSL (masked_features, masked_frames) or add "
            "--param training.temporal_subsampling_size=1.0")
    aug = out["augmentations"]
    if aug.get("augmentations") and "switch" in aug["augmentations"]:
        raise ValueError("'switch' augmentation is not allowed (swaps male/partner)")
    return out


def preset_tag(presets: Sequence[str]) -> str:
    """Episode-name suffix for presets, in a fixed order: '_aug_ssl'."""
    return "".join(f"_{p}" for p in sorted(set(presets)))


# ---------------------------------------------------------------------------
# 4. Event-aware splits
# ---------------------------------------------------------------------------

def segment_spans(n_frames: int, len_segment: int, overlap: int
                  ) -> List[Tuple[int, int]]:
    """DLC2Action's segment grid for one clip: start=i*step, end=min(+len)."""
    step = len_segment - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than len_segment")
    return [(s, min(s + len_segment, n_frames))
            for s in range(0, n_frames, step)]


def segment_id(video: str, start: int, end: int) -> str:
    return f"{video}---{PAIR_CLIP}---{start}-{end}"


def cluster_bouts(positives: Dict[str, List[Tuple[int, int]]],
                  cluster_gap: int) -> List[Dict]:
    """Group bouts closer than ``cluster_gap`` rows into clusters."""
    clusters = []
    for v in sorted(positives):
        cur = None
        for s, e in sorted(positives[v]):
            if cur and s - cur["end"] <= cluster_gap:
                cur["end"] = max(cur["end"], e)
                cur["bouts"].append((s, e))
            else:
                cur = {"video": v, "start": s, "end": e, "bouts": [(s, e)]}
                clusters.append(cur)
    for i, c in enumerate(clusters):
        c["id"] = i
        c["pos_frames"] = int(sum(e - s for s, e in c["bouts"]))
    return clusters


def make_event_split(
    data_dir: Union[str, Path],
    out_path: Union[str, Path],
    len_segment: int = 128,
    overlap: int = 64,
    scheme: str = "cluster",
    fold: int = 0,
    val_cluster: Optional[int] = None,
    cluster_gap: int = 5000,
    context: int = 9000,
    guard: Optional[int] = None,
    test_videos: Sequence[str] = (),
    val_videos: Sequence[str] = (),
    train_background_frac: float = 1.0,
    near_bout: int = 3000,
    behavior: str = "circling",
    seed: int = 0,
    val_frac: float = 0.25,
) -> Dict:
    """Write a DLC2Action ``file`` split with segment-level entries.

    scheme="cluster" (few videos): bouts are grouped into clusters (gap <=
      ``cluster_gap`` H5 rows). Fold k: test = cluster k, val = cluster
      ``val_cluster`` (default k+1 mod K). Each held-out cluster gets a
      window of +/- ``context`` rows of surrounding background. Segments
      straddling a window edge, or within ``guard`` rows of it (default
      len_segment), are dropped to prevent leakage through overlapping
      segments. Everything else is train.
    scheme="video" (>= 3 videos): whole videos in ``test_videos`` /
      ``val_videos``; all others train.
    scheme="crossvideo" (the team protocol, works with 2 videos): fold k
      tests on the WHOLE video ``sorted(videos)[k]`` (or ``test_videos``)
      and trains on the others. Validation (threshold tuning) is the late
      tail of the training video with the most bouts: its last bout
      clusters, enough for >= ``val_frac`` of the training positives, from
      ``context`` rows before the first of them to the end of that video.

    ``train_background_frac`` < 1 keeps only that fraction of TRAIN
    segments that are further than ``near_bout`` rows from any bout
    (seeded); bout-adjacent background is always kept. Val/test are never
    subsampled.

    A JSON sidecar (``<out>.json``) stores the parameters, the held-out
    windows and per-subset coverage; ``score`` and the preflight read it.
    """
    data_dir = Path(data_dir)
    meta = load_meta(data_dir)
    videos = sorted(meta)
    positives = read_positive_intervals(data_dir, videos, behavior)
    guard = len_segment if guard is None else int(guard)
    rng = random.Random(seed)

    windows = {"val": [], "test": []}   # (video, start, end)
    info = {"scheme": scheme}
    if scheme == "cluster":
        clusters = cluster_bouts(positives, cluster_gap)
        K = len(clusters)
        if K < 2:
            raise ValueError(f"need >= 2 bout clusters for scheme 'cluster', "
                             f"found {K}; lower --cluster-gap")
        if not 0 <= fold < K:
            raise ValueError(f"fold must be in 0..{K - 1}")
        v_id = (fold + 1) % K if val_cluster is None else int(val_cluster)
        if v_id == fold or not 0 <= v_id < K:
            raise ValueError("val cluster must differ from the test cluster")
        for subset, cid in (("test", fold), ("val", v_id)):
            c = clusters[cid]
            n = meta[c["video"]]["n_frames"]
            windows[subset].append((c["video"], max(0, c["start"] - context),
                                    min(n, c["end"] + context)))
        info.update({"n_clusters": K, "fold": fold, "val_cluster": v_id,
                     "clusters": [{k: c[k] for k in ("id", "video", "start",
                                                     "end", "pos_frames")}
                                  | {"n_bouts": len(c["bouts"])}
                                  for c in clusters]})
    elif scheme == "video":
        unknown = (set(test_videos) | set(val_videos)) - set(videos)
        if unknown:
            raise ValueError(f"unknown videos {sorted(unknown)}; have {videos}")
        if not test_videos or not val_videos:
            raise ValueError("scheme 'video' needs --test-videos and "
                             "--val-videos")
        if set(test_videos) & set(val_videos):
            raise ValueError("a video cannot be both val and test")
        for subset, vids in (("test", test_videos), ("val", val_videos)):
            for v in vids:
                windows[subset].append((v, 0, meta[v]["n_frames"]))
        info.update({"test_videos": list(test_videos),
                     "val_videos": list(val_videos)})
    elif scheme == "crossvideo":
        if test_videos:
            test_v = list(test_videos)
        elif 0 <= fold < len(videos):
            test_v = [videos[fold]]
        else:
            raise ValueError(f"fold must be in 0..{len(videos) - 1} (one per video)")
        unknown = set(test_v) - set(videos)
        if unknown:
            raise ValueError(f"unknown videos {sorted(unknown)}; have {videos}")
        train_v = [v for v in videos if v not in test_v]
        if not train_v:
            raise ValueError("crossvideo needs at least one training video")
        for v in test_v:
            windows["test"].append((v, 0, int(meta[v]["n_frames"])))
        tpos = {v: positives[v] for v in train_v}
        vv = max(train_v, key=lambda v: sum(e - s for s, e in tpos[v]))
        clusters = cluster_bouts({vv: tpos[vv]}, cluster_gap)
        if len(clusters) < 2:
            raise ValueError(f"{vv} has {len(clusters)} bout cluster(s); need >= 2 "
                             "to hold out a validation tail (lower --cluster-gap)")
        total = sum(e - s for v in train_v for s, e in tpos[v])
        picked, got = [], 0
        for c in sorted(clusters, key=lambda c: c["start"], reverse=True):
            if got >= val_frac * total or len(picked) == len(clusters) - 1:
                break
            picked.append(c)
            got += c["pos_frames"]
        v0 = max(0, min(c["start"] for c in picked) - context)
        windows["val"].append((vv, v0, int(meta[vv]["n_frames"])))
        info.update({"fold": int(fold), "test_videos": test_v,
                     "train_videos": train_v, "val_video": vv,
                     "val_frac_target": val_frac,
                     "val_pos_frac": round(got / max(total, 1), 3),
                     "val_clusters": len(picked), "train_clusters": len(clusters) - len(picked)})
    else:
        raise ValueError("scheme must be 'cluster', 'video' or 'crossvideo'")

    for tv, ts, te in windows["test"]:
        for vv, vs, ve in windows["val"]:
            if tv == vv and vs < te + guard and ts < ve + guard:
                raise ValueError(
                    f"val and test windows overlap in {tv} ({vs}-{ve} vs "
                    f"{ts}-{te}); reduce --context or pick another "
                    "--val-cluster")

    subsets = {"train": [], "val": [], "test": []}
    coverage = {k: {"segments": 0, "pos_frames": 0, "frames": 0,
                    "bouts": 0} for k in subsets}
    excluded = 0
    for v in videos:
        n = int(meta[v]["n_frames"])
        pos = np.zeros(n, dtype=bool)
        for s, e in positives[v]:
            pos[s:e] = True
        near = np.zeros(n + 1, dtype=np.int64)  # prefix sums for "near bout"
        wide = np.zeros(n, dtype=bool)
        for s, e in positives[v]:
            wide[max(0, s - near_bout):min(n, e + near_bout)] = True
        near[1:] = np.cumsum(wide)
        cover = {k: np.zeros(n, dtype=bool) for k in subsets}
        for s, e in segment_spans(n, len_segment, overlap):
            target = "train"
            for subset in ("test", "val"):
                for wv, ws, we in windows[subset]:
                    if wv != v:
                        continue
                    if s >= ws and e <= we:
                        target = subset
                    elif e > ws - guard and s < we + guard:
                        target = None if target == "train" else target
            if target is None:
                excluded += 1
                continue
            if target == "train" and train_background_frac < 1.0:
                if near[e] - near[s] == 0 and \
                        rng.random() >= train_background_frac:
                    excluded += 1
                    continue
            subsets[target].append(segment_id(v, s, e))
            cover[target][s:e] = True
            coverage[target]["segments"] += 1
        for k in subsets:
            coverage[k]["pos_frames"] += int((cover[k] & pos).sum())
            coverage[k]["frames"] += int(cover[k].sum())
            coverage[k]["bouts"] += sum(
                1 for s, e in positives[v] if cover[k][s:e].mean() >= 0.5)

    for k in ("train", "val", "test"):
        if coverage[k]["segments"] == 0:
            raise ValueError(f"split has no {k} segments")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for header, key in (("Training videos:", "train"),
                            ("Validation videos:", "val"),
                            ("Test videos:", "test")):
            f.write(header + "\n")
            for sid in subsets[key]:
                f.write(sid + "\n")
    sidecar = {"split_file": str(out_path.resolve()),
               "len_segment": len_segment, "overlap": overlap,
               "behavior": behavior, "guard": guard, "context": context,
               "cluster_gap": cluster_gap, "val_frac": val_frac,
               "train_background_frac": train_background_frac,
               "near_bout": near_bout, "seed": seed,
               "windows": {k: [list(w) for w in ws]
                           for k, ws in windows.items()},
               "coverage": coverage, "n_excluded_segments": excluded,
               "n_frames": {v: int(meta[v]["n_frames"]) for v in videos},
               **info}
    Path(str(out_path) + ".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return sidecar


def read_split_file(path: Union[str, Path]) -> Dict[str, List[str]]:
    """Parse a DLC2Action split file (same rules as DLC2Action)."""
    out = {"train": [], "val": [], "test": []}
    active = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("Train"):
            active = out["train"]
        elif line.startswith("Valid"):
            active = out["val"]
        elif line.startswith("Test"):
            active = out["test"]
        else:
            s = line.rstrip(",\n ")
            if not s:
                continue
            if active is None:
                raise ValueError(f"{path}: entry before any header: {s!r}")
            active += s.split(", ") if ", " in s else [s]
    return out


def split_coverage_masks(split_path, n_frames: Dict[str, int]
                         ) -> Dict[str, Dict[str, np.ndarray]]:
    """Per subset and video, boolean mask of H5 rows covered by segments."""
    parsed = read_split_file(split_path)
    masks = {k: {v: np.zeros(n, dtype=bool) for v, n in n_frames.items()}
             for k in parsed}
    for k, entries in parsed.items():
        for sid in entries:
            v, clip, rng_ = sid.split("---")
            s, e = map(int, rng_.split("-"))
            if v in masks[k]:
                masks[k][v][s:e] = True
    return masks


# ---------------------------------------------------------------------------
# 5. Predictions -> per-frame probabilities -> events
# ---------------------------------------------------------------------------

def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)


def saved_row_for_behavior(classes: Dict, behavior: str) -> int:
    """Row of a saved prediction array that holds ``behavior``.

    DLC2Action 1.0 ``_make_prediction`` reorders the model output with
    ``new = old[:, perm]``, ``perm[i] = list(classes.values()).index(
    classes[i])``, and model column j is class ``classes[j]``. Because the
    dict is built as ``{1: 'circling', 0: 'other'}`` (insertion order), row 0
    of the saved array is CIRCLING, not "other". Replicate it exactly.
    """
    cd = {int(k): str(v) for k, v in dict(classes).items()}
    if sorted(cd) != list(range(len(cd))):
        raise ValueError(f"unexpected class dict {classes}")
    values = list(dict(classes).values())
    values = [str(v) for v in values]
    for i in range(len(cd)):
        if cd[values.index(cd[i])] == behavior:
            return i
    raise ValueError(f"behavior {behavior!r} not in classes {classes}")


def load_prediction_probs(prediction: Union[str, Path],
                          behavior: str = "circling",
                          behavior_index: Optional[int] = None,
                          ) -> Dict[str, np.ndarray]:
    """Per-video P(behavior) on the H5 row axis from DLC2Action predictions.

    ``prediction`` is a prediction folder
    (``results/predictions/<name>/``) or one ``*_prediction.pickle`` file.
    Clip arrays are placed at their ``min_frames`` offset; uncovered frames
    (-100) become 0. Multiple clips are combined with max.
    """
    prediction = Path(prediction)
    if prediction.is_dir():
        files = sorted(prediction.glob("*_prediction.pickle"))
        name = prediction.name
        if not files:
            raise FileNotFoundError(f"no *_prediction.pickle in {prediction}")
    else:
        files = [prediction]
        name = prediction.parent.name
    out = {}
    for f in files:
        suffix = f"_{name}_prediction.pickle"
        video = (f.name[:-len(suffix)] if f.name.endswith(suffix)
                 else f.name.replace("_prediction.pickle", ""))
        with open(f, "rb") as fh:
            res = pickle.load(fh)
        if not isinstance(res, dict):
            raise ValueError(f"{f}: unexpected content {type(res)}")
        classes = res.get("classes") or {}
        min_frames = res.get("min_frames") or {}
        idx = behavior_index
        if idx is None:
            if not classes:
                raise ValueError(f"{f}: no 'classes' in the pickle; pass "
                                 "behavior_index explicitly")
            idx = saved_row_for_behavior(classes, behavior)
        tracks = []
        for clip, arr in res.items():
            if clip in ("classes", "min_frames", "max_frames"):
                continue
            a = _to_numpy(arr)
            if a.ndim != 2:
                continue
            p = a[idx].copy()
            p[p < 0] = 0.0                       # -100 = not covered
            off = int(min_frames.get(clip, 0))
            tracks.append((off, p))
        if not tracks:
            logger.warning(f"{f}: no clip arrays found")
            continue
        n = max(off + len(p) for off, p in tracks)
        full = np.zeros(n)
        for off, p in tracks:
            full[off:off + len(p)] = np.maximum(full[off:off + len(p)], p)
        out[video] = full
    return out


def probs_to_intervals(probs: np.ndarray, threshold: float,
                       min_frames: int = 1, gap_frames: int = 0
                       ) -> List[Tuple[int, int]]:
    """Threshold -> [start, end) intervals, merging gaps <= gap_frames."""
    on = probs > threshold
    edges = np.diff(np.concatenate([[0], on.astype(np.int8), [0]]))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    ivs = list(zip(starts.tolist(), ends.tolist()))
    if gap_frames > 0 and ivs:
        merged = [list(ivs[0])]
        for s, e in ivs[1:]:
            if s - merged[-1][1] <= gap_frames:
                merged[-1][1] = e
            else:
                merged.append([s, e])
        ivs = [tuple(x) for x in merged]
    return [(s, e) for s, e in ivs if e - s >= max(1, min_frames)]


def predictions_to_events(
    prediction: Union[str, Path],
    meta_path: Optional[Union[str, Path]] = None,
    behavior: str = "circling",
    threshold: float = 0.5,
    min_duration_s: float = 1.0,
    min_gap_s: float = 0.0,
    fps: Optional[float] = None,
    behavior_index: Optional[int] = None,
) -> pd.DataFrame:
    """DLC2Action prediction folder -> events DataFrame (shared schema).

    Frames are mapped back to the ORIGINAL video axis with
    ``orig = h5_row * stride + frame_min`` using ``_conversion_meta.json``.
    ``fps`` overrides the effective (post-stride) fps from the metadata.
    """
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8")) if meta_path else {}
    probs = load_prediction_probs(prediction, behavior, behavior_index)
    events = []
    for video, p in sorted(probs.items()):
        m = meta.get(video, {})
        stride = int(m.get("stride", 1))
        fmin = int(m.get("frame_min", 0))
        vfps = fps or m.get("fps")
        min_f = int(round(min_duration_s * vfps)) if vfps else 1
        gap_f = int(round(min_gap_s * vfps)) if vfps else 0
        for k, (s, e) in enumerate(probs_to_intervals(p, threshold, min_f,
                                                      gap_f), 1):
            s_o = s * stride + fmin
            e_o = (e - 1) * stride + fmin
            fps_full = (vfps * stride) if vfps else None
            events.append({
                "video": video,
                "event_id": f"{video}_dlc2action_evt{k:03d}",
                "detection_status": "detected",
                "start_frame": int(s_o), "end_frame": int(e_o),
                "start_time_s": round(s_o / fps_full, 3) if fps_full else np.nan,
                "end_time_s": round(e_o / fps_full, 3) if fps_full else np.nan,
                "duration_s": round((e - s) / vfps, 3) if vfps else np.nan,
                "detection_confidence": round(float(p[s:e].mean()), 4),
                "notes": f"DLC2Action prediction ({behavior})",
            })
    cols = ["video", "event_id", "detection_status", "start_frame",
            "end_frame", "start_time_s", "end_time_s", "duration_s",
            "detection_confidence", "notes"]
    return pd.DataFrame(events, columns=cols)


# ---------------------------------------------------------------------------
# 6. Scoring on held-out windows
# ---------------------------------------------------------------------------

def _frame_counts(p, gt, mask, thr):
    pred = p > thr
    tp = int((pred & gt & mask).sum())
    fp = int((pred & ~gt & mask).sum())
    fn = int((~pred & gt & mask).sum())
    return tp, fp, fn


def _prf(tp, fp, fn):
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (2 * prec * rec / (prec + rec)
          if prec == prec and rec == rec and prec + rec else float("nan"))
    return prec, rec, f1


def score_predictions(
    prediction: Union[str, Path],
    data_dir: Union[str, Path],
    split_path: Union[str, Path],
    behavior: str = "circling",
    threshold: Optional[float] = None,
    min_duration_s: float = 1.0,
    min_gap_s: float = 0.0,
    iou: float = 0.3,
    fps: Optional[float] = None,
) -> Dict:
    """Frame- and event-level metrics on the val and test parts of a split.

    Only frames covered by val/test segments are scored (train frames are
    ignored, guard-band frames too). Frames annotated ``unknown`` are
    excluded. If ``threshold`` is None it is chosen on VAL (max event F1,
    ties -> frame F1) and then applied unchanged to TEST.

    Event matching: a GT bout counts as detected if a predicted event
    overlaps it with IoU >= ``iou`` (one-to-one, greedy). Unmatched
    predicted events are false positives; reported per hour of scored
    background.
    """
    data_dir = Path(data_dir)
    meta = load_meta(data_dir)
    probs = load_prediction_probs(prediction, behavior)
    n_frames = {v: int(m["n_frames"]) for v, m in meta.items()}
    masks = split_coverage_masks(split_path, n_frames)
    pos = read_positive_intervals(data_dir, list(meta), behavior)
    ign = read_ignored_intervals(data_dir, list(meta))

    def gt_arr(v):
        a = np.zeros(n_frames[v], dtype=bool)
        for s, e in pos[v]:
            a[s:e] = True
        return a

    def evaluate(subset, thr):
        tp = fp = fn = 0
        ev_tp = ev_fn = ev_fp = 0
        bg_frames = 0
        fps_used = None
        for v in sorted(meta):
            if v not in probs:
                continue
            mask = masks[subset][v].copy()
            for s, e in ign[v]:
                mask[s:e] = False
            if not mask.any():
                continue
            p = np.zeros(n_frames[v])
            q = probs[v][:n_frames[v]]
            p[:len(q)] = q
            g = gt_arr(v)
            a, b, c = _frame_counts(p, g, mask, thr)
            tp, fp, fn = tp + a, fp + b, fn + c
            vfps = fps or meta[v].get("fps")
            fps_used = vfps
            min_f = int(round(min_duration_s * vfps)) if vfps else 1
            gap_f = int(round(min_gap_s * vfps)) if vfps else 0
            pred_iv = [(s, e) for s, e in probs_to_intervals(
                p * mask, thr, min_f, gap_f) if mask[s:e].mean() >= 0.5]
            gt_iv = [(s, e) for s, e in pos[v] if mask[s:e].mean() >= 0.5]
            used = set()
            for s, e in gt_iv:
                best, bj = 0.0, None
                for j, (ps, pe) in enumerate(pred_iv):
                    if j in used:
                        continue
                    inter = max(0, min(e, pe) - max(s, ps))
                    union = max(e, pe) - min(s, ps)
                    if union and inter / union > best:
                        best, bj = inter / union, j
                if bj is not None and best >= iou:
                    used.add(bj)
                    ev_tp += 1
                else:
                    ev_fn += 1
            ev_fp += len(pred_iv) - len(used)
            bg_frames += int((mask & ~g).sum())
        if tp + fn + fp == 0 and ev_tp + ev_fn == 0:
            return None
        fprec, frec, ff1 = _prf(tp, fp, fn)
        eprec, erec, ef1 = _prf(ev_tp, ev_fp, ev_fn)
        hours = (bg_frames / fps_used / 3600) if fps_used else float("nan")
        return {"threshold": thr,
                "frame_precision": fprec, "frame_recall": frec,
                "frame_f1": ff1, "frame_tp": tp, "frame_fp": fp,
                "frame_fn": fn,
                "event_precision": eprec, "event_recall": erec,
                "event_f1": ef1, "bouts_detected": ev_tp,
                "bouts_missed": ev_fn, "false_events": ev_fp,
                "background_hours": hours,
                "false_events_per_hour": (ev_fp / hours
                                          if hours and hours == hours
                                          else float("nan"))}

    result = {"chosen_on": "fixed" if threshold is not None else "val"}
    if threshold is None:
        best = None
        for thr in np.round(np.arange(0.05, 0.96, 0.05), 2):
            r = evaluate("val", float(thr))
            if r is None:
                continue
            key = (np.nan_to_num(r["event_f1"]), np.nan_to_num(r["frame_f1"]))
            if best is None or key > best[0]:
                best = (key, float(thr))
        threshold = best[1] if best else 0.5
    result["threshold"] = threshold
    result["val"] = evaluate("val", threshold)
    result["test"] = evaluate("test", threshold)
    return result
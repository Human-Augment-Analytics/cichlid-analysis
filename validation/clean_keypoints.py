"""
clean_keypoints.py: write a cleaned copy of <video>_vid.parquet.

The output has the same filename, columns, types and metadata as the input, plus qc_* flag columns, and is
written to a separate folder so every existing loader can read it by pointing --data-dir there.

What it does, in order (each check only flags; all edits happen in apply_flags):
  1. single-row checks   reflection (box center off the gravel outline in floors.json),
                         coincident keypoints (informational only)
  2. same-frame checks   one shared table of nearby detections (close_pairs), then:
                         double detection (boxes overlap + same skeleton -> one row dropped),
                         copied skeleton (identical keypoints on separate boxes -> the row whose box does not
                         contain the skeleton is flagged), box overlap (two different fish overlapping:
                         information only), keypoint jump (skeleton center outside its own box)
  3. over-time checks    body-length jumps, run per track on rows that passed 1-2
  4. apply               delete flagged rows: double detections, reflections (unless --reflections flag), and
                         rows with a bad pose (copied skeleton, keypoint jump, body-length jump); with
                         --bad-pose nan those last rows are kept instead, pose set to NaN and position = box center.
                         Then recompute the columns that depend on neighbouring rows.
  4b. new columns        sex_track_vote: majority Sex label of each track (the Sex column itself is not changed);
                         track_chain_id: tracks glued across short, unambiguous breaks (TrackUID not changed)
  5. write + check       same schema as the input, metadata + cleaned_by note, contract check, summary CSV, and
                         <video>_vid_deleted.parquet: every deleted row, unchanged, with its qc_* flags and a
                         deleted_reason column, so anyone can see what was taken out.

Turning rate and speed are recomputed as the pipeline does across missed frames: from the last valid row of the
track within 0.2 s, divided by the frames elapsed; is_track_start / is_track_end are rebuilt after rows are dropped.

Usage
  python validation/clean_keypoints.py 0031 --data-dir C:\\...\\Downloads --out-dir C:\\...\\Downloads\\clean
  python validation/clean_keypoints.py 0031 0028 0013 --data-dir ... --out-dir ... --pairs   (also builds pair files)
"""
import argparse
import json
import pathlib

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from matplotlib.path import Path as Polygon

HERE = pathlib.Path(__file__).resolve().parent
FLOORS_FILE = HERE / "floors.json"
FPS = 30.0

KEYPOINTS = ["Nose", "LeftEye", "RightEye", "Head", "Spine1", "Spine2", "Spine3", "Spine4", "Peduncle", "TailTip"]
KP_COLS = [f"{k}_{a}" for k in KEYPOINTS for a in ("x", "y")]
ORIENTATION_COLS = ["heading_rad", "body_axis_rad", "bend_angle_rad"]

# thresholds (pixels unless stated)
DUPLICATE_PX = 40          # two detections whose box centers are closer than this may be one fish
DUPLICATE_IOU = 0.5        # ... and whose boxes overlap this much (intersection / union) are the same box
SAME_FISH_BODY_LENGTHS = 0.5   # ... and whose skeletons differ by at most this much are one fish detected twice
COPY_PX = 0.5              # two skeletons this close on every keypoint are copies of each other
GAP_FRAMES = 6             # rates bridge at most this many frames (0.2 s), as the pipeline does
LENGTH_RATIO = (0.6, 1.6)  # body length outside this ratio of the track's running median = bad pose
LENGTH_WINDOW = 15         # rows in that running median
LINK_GAP_FRAMES = 10       # glue track B onto track A if B starts at most this many frames after A ends,
LINK_BODY_LENGTHS = 0.5    # ... within this many median body lengths of where A ended, and the match is one-to-one

FLAG_COLS = ["qc_reflection", "qc_double_detection", "qc_copied_skeleton", "qc_keypoint_jump",
             "qc_length_jump", "qc_coincident_kp", "qc_box_overlap"]


# ---------- 1. load ----------------------------------------------------------------------------------------
def load_fish(data_dir, video) -> tuple[pd.DataFrame, dict]:
    path = pathlib.Path(data_dir) / f"{video}_vid.parquet"
    df = pd.read_parquet(path)
    metadata = pq.read_schema(path).metadata or {}
    return df, metadata


def load_floor(video, floors_file=FLOORS_FILE) -> list:
    floors = json.loads(pathlib.Path(floors_file).read_text())
    if video not in floors:
        raise SystemExit(f"No floor outline for {video} in {floors_file}. "
                         f"Draw one with: python validation/draw_floor.py {video} --videos <folder>")
    return floors[video]


# ---------- helpers ----------------------------------------------------------------------------------------
def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def box_bounds(df):
    return (df.X_center - df.Width / 2, df.X_center + df.Width / 2,
            df.Y_center - df.Height / 2, df.Y_center + df.Height / 2)


def skeleton_in_box(cx, cy, box_x, box_y, width, height):
    """True where the point (cx, cy), a skeleton center, lies inside the box centered at (box_x, box_y).
    Used to decide which box a copied skeleton belongs to and whether a row's own skeleton left its box."""
    return ((cx - box_x).abs() < width / 2) & ((cy - box_y).abs() < height / 2)


def same_frame_pairs(fish, cols):
    """Every two rows of the same frame (left row index < right row index), with the requested columns."""
    f = fish[["FrameNum"] + cols].assign(row=fish.index.to_numpy())
    m = f.merge(f, on="FrameNum", suffixes=("_a", "_b"))
    return m[m.row_a < m.row_b].reset_index(drop=True)


# ---------- 2. single-row checks ---------------------------------------------------------------------------
def flag_reflection(fish, floor) -> pd.Series:
    inside = Polygon(floor).contains_points(fish[["X_center", "Y_center"]].to_numpy())
    return pd.Series(~inside, index=fish.index)


def flag_coincident_keypoints(fish) -> pd.Series:
    return fish.qc_n_coincident_kp.fillna(0) > 0


# ---------- 3. same-frame checks ---------------------------------------------------------------------------
def close_pairs(fish, track_rows) -> pd.DataFrame:
    """Every two detections of a frame that could be the same fish or share a skeleton, with everything the
    same-frame checks need computed once:
      candidates   box centers < DUPLICATE_PX apart (does not depend on keypoints), or skeleton centers
                   < COPY_PX apart (copies, whose boxes can be far apart, e.g. on a reflection)
      iou          box overlap: intersection area / union area (1 = same box, 0 = not touching)
      kp_gap       largest distance between matching keypoints (0 = identical skeletons)
      a_has, b_has whether each box contains the skeleton center of row a (the shared center for copies)
      len_a, len_b rows in each detection's track over the whole video"""
    cols = ["X_center", "Y_center", "Width", "Height", "midline_centroid_x", "midline_centroid_y", "TrackUID"] + KP_COLS
    m = same_frame_pairs(fish, cols)
    box_d = np.hypot(m.X_center_a - m.X_center_b, m.Y_center_a - m.Y_center_b)
    skel_d = np.hypot(m.midline_centroid_x_a - m.midline_centroid_x_b, m.midline_centroid_y_a - m.midline_centroid_y_b)
    m = m[(box_d < DUPLICATE_PX) | (skel_d < COPY_PX)].reset_index(drop=True)

    ax0, ax1 = m.X_center_a - m.Width_a / 2, m.X_center_a + m.Width_a / 2
    ay0, ay1 = m.Y_center_a - m.Height_a / 2, m.Y_center_a + m.Height_a / 2
    bx0, bx1 = m.X_center_b - m.Width_b / 2, m.X_center_b + m.Width_b / 2
    by0, by1 = m.Y_center_b - m.Height_b / 2, m.Y_center_b + m.Height_b / 2
    inter = ((np.minimum(ax1, bx1) - np.maximum(ax0, bx0)).clip(lower=0)
             * (np.minimum(ay1, by1) - np.maximum(ay0, by0)).clip(lower=0))
    iou = inter / (m.Width_a * m.Height_a + m.Width_b * m.Height_b - inter)

    per_keypoint = np.stack([np.hypot(m[f"{k}_x_a"] - m[f"{k}_x_b"], m[f"{k}_y_a"] - m[f"{k}_y_b"]) for k in KEYPOINTS])
    cx, cy = m.midline_centroid_x_a, m.midline_centroid_y_a
    return pd.DataFrame({
        "row_a": m.row_a, "row_b": m.row_b, "iou": iou,
        "kp_gap": np.nanmax(per_keypoint, axis=0),
        "a_has": skeleton_in_box(cx, cy, m.X_center_a, m.Y_center_a, m.Width_a, m.Height_a),
        "b_has": skeleton_in_box(cx, cy, m.X_center_b, m.Y_center_b, m.Width_b, m.Height_b),
        "len_a": track_rows.reindex(m.TrackUID_a).to_numpy(),
        "len_b": track_rows.reindex(m.TrackUID_b).to_numpy(),
    })


def _rows(fish, rows) -> pd.Series:
    out = pd.Series(False, index=fish.index)
    out.loc[np.unique(np.asarray(rows))] = True
    return out


def flag_double_detection(fish, pairs) -> pd.Series:
    """One fish detected twice: boxes overlap (IoU > DUPLICATE_IOU) and the skeletons are the same or nearly the
    same (worst keypoint within SAME_FISH_BODY_LENGTHS body lengths). The row to drop: for identical skeletons
    the row whose box does not contain it; otherwise (or if both do) the row from the shorter track."""
    same_fish = pairs.kp_gap <= SAME_FISH_BODY_LENGTHS * fish.body_length_px.median()
    p = pairs[(pairs.iou > DUPLICATE_IOU) & same_fish]
    shorter = np.where(p.len_a < p.len_b, p.row_a, p.row_b)
    owner_known = (p.kp_gap < COPY_PX) & (p.a_has != p.b_has)
    not_owner = np.where(p.a_has, p.row_b, p.row_a)
    return _rows(fish, np.where(owner_known, not_owner, shorter))


def flag_copied_skeleton(fish, pairs) -> pd.Series:
    """Identical skeletons on two detections whose boxes do NOT overlap much (those are double detections).
    The row whose box contains the skeleton center keeps it; the other is flagged. If both boxes contain it,
    it is one fish with two shifted boxes: the row from the shorter track is flagged, as for double detections.
    If neither box contains it, both rows are flagged."""
    p = pairs[(pairs.kp_gap < COPY_PX) & (pairs.iou <= DUPLICATE_IOU)]
    clear = (p.a_has != p.b_has).to_numpy()
    both = (p.a_has & p.b_has).to_numpy()
    neither = ~clear & ~both
    shorter = np.where(p.len_a < p.len_b, p.row_a, p.row_b)
    flagged = np.concatenate([np.where(p.a_has, p.row_b, p.row_a)[clear], shorter[both],
                              p.row_a[neither], p.row_b[neither]])
    return _rows(fish, flagged)


def flag_box_overlap(fish, pairs) -> pd.Series:
    """Information only: boxes overlap heavily but the skeletons are clearly different, e.g. one fish swimming
    over another. Both rows are kept as they are; their poses are still checked by the other rules."""
    same_fish = pairs.kp_gap <= SAME_FISH_BODY_LENGTHS * fish.body_length_px.median()
    p = pairs[(pairs.iou > DUPLICATE_IOU) & ~same_fish]
    return _rows(fish, np.concatenate([p.row_a, p.row_b]))


def flag_keypoint_jump(fish) -> pd.Series:
    """The row's own skeleton center lies outside its own box: the pose belongs to something else.
    Rows without a skeleton are not flagged."""
    inside = skeleton_in_box(fish.midline_centroid_x, fish.midline_centroid_y,
                             fish.X_center, fish.Y_center, fish.Width, fish.Height)
    return ~inside & fish.midline_centroid_x.notna()


# ---------- 4. over-time checks (per track, on rows that passed the checks above) --------------------------
def _track_order(fish, usable):
    f = fish.loc[usable, ["TrackUID", "FrameNum", "heading_rad", "midline_centroid_x", "midline_centroid_y",
                          "body_length_px"]]
    return f.sort_values(["TrackUID", "FrameNum"])


def flag_length_jumps(fish, usable) -> pd.Series:
    """Body length outside LENGTH_RATIO of the track's running median (LENGTH_WINDOW rows)."""
    f = _track_order(fish, usable)
    med = f.groupby("TrackUID").body_length_px.transform(
        lambda s: s.rolling(LENGTH_WINDOW, center=True, min_periods=1).median())
    ratio = f.body_length_px / med
    out = pd.Series(False, index=fish.index)
    out.loc[f.index[((ratio < LENGTH_RATIO[0]) | (ratio > LENGTH_RATIO[1])).to_numpy()]] = True
    return out


# ---------- 5. apply ---------------------------------------------------------------------------------------
DELETE_REASONS = ["qc_double_detection", "qc_reflection", "qc_copied_skeleton", "qc_keypoint_jump", "qc_length_jump"]


def apply_flags(fish, flags, drop_reflections=True, drop_bad_pose=True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (clean, deleted). deleted holds the removed rows unchanged, plus their flags and deleted_reason
    (the first reason in DELETE_REASONS order when a row has several)."""
    bad_pose = flags.qc_copied_skeleton | flags.qc_keypoint_jump | flags.qc_length_jump
    drop = flags.qc_double_detection | (flags.qc_reflection if drop_reflections else False)
    if drop_bad_pose:
        drop = drop | bad_pose
    both = pd.concat([fish, flags], axis=1)
    deleted = both.loc[drop].copy()
    reasons = [r for r in DELETE_REASONS if r != "qc_reflection" or drop_reflections]
    if not drop_bad_pose:
        reasons = [r for r in reasons if r in ("qc_double_detection", "qc_reflection")]
    deleted["deleted_reason"] = np.select([deleted[r] for r in reasons], [r[3:] for r in reasons], default="")
    clean = both.loc[~drop].copy()
    wrong_fish = clean.qc_copied_skeleton | clean.qc_keypoint_jump | clean.qc_length_jump   # empty when deleted
    no_pose = wrong_fish
    clean.loc[no_pose, KP_COLS + ORIENTATION_COLS] = np.nan
    clean.loc[wrong_fish, "body_length_px"] = np.nan
    clean.loc[wrong_fish, "midline_centroid_x"] = clean.loc[wrong_fish, "X_center"]
    clean.loc[wrong_fish, "midline_centroid_y"] = clean.loc[wrong_fish, "Y_center"]
    clean.loc[no_pose, "qc_n_missing_kp"] = len(KEYPOINTS)
    clean["qc_pose_ok"] = ~no_pose
    return clean, deleted


def _last_valid(values, valid, frames, tracks):
    """For every row: the value and frame of the most recent EARLIER row of the same track where valid is True
    (NaN if there is none)."""
    v = values.where(valid)
    f = frames.where(valid).astype("float64")
    prev_v = v.groupby(tracks).shift().groupby(tracks).ffill()
    prev_f = f.groupby(tracks).shift().groupby(tracks).ffill()
    return prev_v, prev_f


def recompute_derived(clean, original) -> pd.DataFrame:
    """Recompute the columns that depend on neighbouring rows of a track, for every row, with the pipeline's own
    convention (checked against the delivered files): a rate is taken from the most recent earlier row of the
    same track that has a valid value, divided by the frames elapsed, and is NaN if that row is more than
    GAP_FRAMES (0.2 s) back. Removed poses and dropped rows are therefore treated like missed detections.
      d_heading_rad_s  wrap(heading - previous valid heading) / frames elapsed * fps
      speed_bl_s       distance between skeleton centers / frames elapsed * fps / the track's median body length
                       (rows whose position was replaced by the box center are left out of the speed chain)
      is_track_start / is_track_end   first / last remaining row of each track"""
    c = clean.sort_values(["TrackUID", "FrameNum"])
    tracks, frames = c.TrackUID, c.FrameNum

    heading = c.heading_rad.astype("float64")
    prev_h, prev_hf = _last_valid(heading, heading.notna(), frames, tracks)
    step_h = frames - prev_hf
    d_heading = (wrap(heading - prev_h) / step_h * FPS).where(step_h <= GAP_FRAMES)

    own_center = pd.Series(True, index=c.index)
    for col in ("qc_copied_skeleton", "qc_keypoint_jump", "qc_length_jump"):
        if col in c:
            own_center &= ~c[col]
    x, y = c.midline_centroid_x.astype("float64"), c.midline_centroid_y.astype("float64")
    prev_x, prev_f = _last_valid(x, own_center, frames, tracks)
    prev_y, _ = _last_valid(y, own_center, frames, tracks)
    step_p = frames - prev_f
    track_len = original.groupby("TrackUID").body_length_px.median()
    speed = (np.hypot(x - prev_x, y - prev_y) / step_p * FPS / tracks.map(track_len))
    speed = speed.where(own_center & (step_p <= GAP_FRAMES))

    c["d_heading_rad_s"] = d_heading.astype(original.d_heading_rad_s.dtype)
    c["speed_bl_s"] = speed.astype(original.speed_bl_s.dtype)
    c["is_track_start"] = ~tracks.duplicated(keep="first")
    c["is_track_end"] = ~tracks.duplicated(keep="last")
    return c.loc[clean.index]


def link_tracks(clean) -> pd.Series:
    """track_chain_id: tracks glued across short breaks. Track B is attached to track A when B's first row is
    1..LINK_GAP_FRAMES frames after A's last row and within LINK_BODY_LENGTHS median body lengths of it, and
    A is B's only such candidate and B is A's only one. A chain is named after its first TrackUID.
    TrackUID itself is not changed; linked tracks never overlap in time, so (FrameNum, track_chain_id) is unique."""
    c = clean.sort_values(["TrackUID", "FrameNum"])
    cols = ["TrackUID", "FrameNum", "midline_centroid_x", "midline_centroid_y"]
    ends = c.groupby("TrackUID").tail(1)[cols]
    starts = c.groupby("TrackUID").head(1)[cols]
    cand = pd.concat([starts.assign(FrameNum=starts.FrameNum - k).merge(ends, on="FrameNum", suffixes=("_s", "_e"))
                      for k in range(1, LINK_GAP_FRAMES + 1)])
    dist = np.hypot(cand.midline_centroid_x_s - cand.midline_centroid_x_e, cand.midline_centroid_y_s - cand.midline_centroid_y_e)
    cand = cand[dist <= LINK_BODY_LENGTHS * clean.body_length_px.median()]
    one_to_one = (cand.groupby("TrackUID_s").TrackUID_s.transform("size").eq(1)
                  & cand.groupby("TrackUID_e").TrackUID_e.transform("size").eq(1))
    parent = pd.Series(cand.TrackUID_e[one_to_one].to_numpy(), index=cand.TrackUID_s[one_to_one].to_numpy())
    root = pd.Series(clean.TrackUID.unique(), index=clean.TrackUID.unique())
    while True:  # follow each track back to the first track of its chain
        step = root.map(parent).fillna(root).astype(root.dtype)
        if step.equals(root):
            break
        root = step
    return clean.TrackUID.map(root).astype(clean.TrackUID.dtype)


def sex_track_vote(clean) -> pd.Series:
    """Majority vote of the Sex label over each track's remaining rows (per-frame labels flip within a track).
    A tie keeps each row's own label. The original Sex column is left unchanged."""
    counts = clean.groupby(["TrackUID", "Sex"]).size().unstack(fill_value=0)
    winner = counts.idxmax(axis=1).where(counts.max(axis=1).gt(counts.sum(axis=1) / 2))
    return clean.TrackUID.map(winner).fillna(clean.Sex).astype(clean.Sex.dtype)


# ---------- 6. write and check -----------------------------------------------------------------------------
def write_clean(clean, original, metadata, out_path, settings) -> pa.Table:
    """Same column order and Arrow types as the input, then the qc_* columns; input metadata + cleaning notes."""
    table = pa.Table.from_pandas(clean.reset_index(drop=True), preserve_index=False)
    src = pa.Table.from_pandas(original.head(1), preserve_index=False).schema
    cols = list(original.columns) + [c for c in clean.columns if c not in original.columns]
    fields = [src.field(c) if c in original.columns else table.schema.field(c) for c in cols]
    table = table.select(cols).cast(pa.schema(fields))
    meta = {k: v for k, v in metadata.items() if k != b"pandas"}
    meta[b"cleaned_by"] = b"cichlid-analysis/validation/clean_keypoints.py"
    meta[b"cleaning_settings"] = json.dumps(settings).encode()
    table = table.replace_schema_metadata(meta)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path)
    return table


def check_contract(out_path, original, metadata) -> None:
    """Fail if the written file would break the team loaders."""
    schema = pq.read_schema(out_path)
    src = pa.Table.from_pandas(original.head(1), preserve_index=False).schema
    problems = [f"column {c}: {schema.field(c).type} != {src.field(c).type}"
                for c in original.columns if c not in schema.names or schema.field(c).type != src.field(c).type]
    meta = schema.metadata or {}
    for key in (b"fps", b"y_axis", b"coordinate_space", b"units"):
        if meta.get(key) != metadata.get(key):
            problems.append(f"metadata {key.decode()}: {meta.get(key)} != {metadata.get(key)}")
    keys = pq.read_table(out_path, columns=["FrameNum", "TrackUID"]).to_pandas()
    if keys.duplicated().any():
        problems.append(f"{int(keys.duplicated().sum())} duplicate (FrameNum, TrackUID) rows")
    if problems:
        raise ValueError(f"{out_path.name} breaks the loader contract:\n  " + "\n  ".join(problems))


def summarize(video, fish, flags, clean, deleted) -> pd.DataFrame:
    rows = [{"video": video, "check": c, "rows": int(flags[c].sum()), "share_of_rows": flags[c].mean()}
            for c in FLAG_COLS]
    for reason, n in deleted.deleted_reason.value_counts().items():
        rows.append({"video": video, "check": f"deleted: {reason}", "rows": int(n), "share_of_rows": n / len(fish)})
    rows.append({"video": video, "check": "rows deleted (total)", "rows": len(fish) - len(clean),
                 "share_of_rows": 1 - len(clean) / len(fish)})
    rows.append({"video": video, "check": "pose removed (rows kept)", "rows": int((~clean.qc_pose_ok).sum()),
                 "share_of_rows": (~clean.qc_pose_ok).sum() / len(fish)})
    return pd.DataFrame(rows)


# ---------- main -------------------------------------------------------------------------------------------
def clean_video(video, data_dir, out_dir, floors_file=FLOORS_FILE, drop_reflections=True, drop_bad_pose=True):
    fish, metadata = load_fish(data_dir, video)
    floor = load_floor(video, floors_file)
    track_rows = fish.groupby("TrackUID").size()

    flags = pd.DataFrame(index=fish.index)
    flags["qc_reflection"] = flag_reflection(fish, floor)
    flags["qc_coincident_kp"] = flag_coincident_keypoints(fish)
    pairs = close_pairs(fish, track_rows)
    flags["qc_double_detection"] = flag_double_detection(fish, pairs)
    flags["qc_copied_skeleton"] = flag_copied_skeleton(fish, pairs) & ~flags.qc_double_detection
    flags["qc_box_overlap"] = flag_box_overlap(fish, pairs)
    flags["qc_keypoint_jump"] = flag_keypoint_jump(fish) & ~flags.qc_copied_skeleton
    usable = ~(flags.qc_reflection | flags.qc_double_detection | flags.qc_copied_skeleton | flags.qc_keypoint_jump)
    flags["qc_length_jump"] = flag_length_jumps(fish, usable)
    flags = flags[FLAG_COLS]

    clean, deleted = apply_flags(fish, flags, drop_reflections, drop_bad_pose)
    assert len(clean) + len(deleted) == len(fish)
    clean = recompute_derived(clean, fish)
    clean["sex_track_vote"] = sex_track_vote(clean)
    clean["track_chain_id"] = link_tracks(clean)
    assert not clean.duplicated(["FrameNum", "track_chain_id"]).any(), "two rows of one chain in the same frame"
    settings = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, tuple))}
    settings["drop_reflections"] = drop_reflections
    settings["drop_bad_pose"] = drop_bad_pose
    out_path = pathlib.Path(out_dir) / f"{video}_vid.parquet"
    write_clean(clean, fish, metadata, out_path, settings)
    check_contract(out_path, fish, metadata)
    deleted_path = out_path.with_name(f"{video}_vid_deleted.parquet")
    deleted.sort_values(["FrameNum", "TrackUID"]).to_parquet(deleted_path, index=False)
    summary = summarize(video, fish, flags, clean, deleted)
    summary.to_csv(out_path.with_name(f"{video}_vid_cleaning_summary.csv"), index=False)
    return out_path, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+", help="video IDs, e.g. 0031 0028")
    ap.add_argument("--data-dir", required=True, help="folder with <video>_vid.parquet")
    ap.add_argument("--out-dir", required=True, help="folder for the cleaned files (not the input folder)")
    ap.add_argument("--floors", default=str(FLOORS_FILE))
    ap.add_argument("--reflections", choices=["drop", "flag"], default="drop",
                    help="drop reflection rows (default) or keep them with qc_reflection=True")
    ap.add_argument("--bad-pose", choices=["drop", "nan"], default="drop",
                    help="delete rows with a bad pose (default) or keep them with the pose set to NaN")
    ap.add_argument("--pairs", action="store_true", help="also build <video>_vid_pairs.parquet in --out-dir")
    args = ap.parse_args()
    if pathlib.Path(args.out_dir).resolve() == pathlib.Path(args.data_dir).resolve():
        raise SystemExit("--out-dir must differ from --data-dir so the originals are never overwritten")
    for video in args.videos:
        out_path, summary = clean_video(video, args.data_dir, args.out_dir, args.floors, args.reflections == "drop",
                                        args.bad_pose == "drop")
        print(summary.to_string(index=False, formatters={"share_of_rows": "{:.2%}".format}))
        print(f"wrote {out_path}")
        if args.pairs:
            from build_pairs import build_pairs, write_pairs
            pairs = build_pairs(pd.read_parquet(out_path))
            pairs_path = out_path.with_name(f"{video}_vid_pairs.parquet")
            write_pairs(pairs, out_path, pairs_path)
            print(f"wrote {pairs_path} ({len(pairs):,} pair rows)")


if __name__ == "__main__":
    main()

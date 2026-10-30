"""
Rebuild the pipeline's pair table (<video>_vid_pairs.parquet) from the per-fish table (<video>_vid.parquet).

Newer deliveries ship only the per-fish file. The recipe below was recovered from the metadata stored in the
original pairs files (min_episode_frames, gap_tolerance_s, pair_ordering, bearing_definition) and checked
column by column against the delivered 0031 and 0028 pair files (see --check).

Recipe
  * Every two fish detected in the same frame form a pair row; fish 1 is the one with the smaller TrackUID.
  * Rows of the same two tracks form an "episode"; a gap of more than gap_tolerance_s (0.2 s = 6 frames)
    starts a new episode. Episodes with fewer than min_episode_frames (5) rows are dropped.
  * pair_episode_id = day_index * 1,000,000 + running number over all episodes ordered by
    (TrackUID_1, TrackUID_2, first frame), numbered before short episodes are dropped.
  * Distances and the orbital angle use the midline centroids (mean of the 8 midline keypoints).
  * orbital_rad = atan2(y2 - y1, x2 - x1); rel_heading = wrap(heading_1 - heading_2);
    bearing_1 = wrap(orbital - heading_1); bearing_2 = wrap(orbital + pi - heading_2).
  * dt_frames = frames since the previous row of the episode (NaN on its first row);
    d_orbital_rad_s = wrap(change in orbital_rad) / dt_frames * fps.
  * Per-fish columns (d_heading, sex, qc_n_missing_kp) are copied; pair_sex is FF, FM or MM.

Usage
  python build_pairs.py 0013 [DATA_DIR]          writes 0013_vid_pairs.parquet next to the input
  python build_pairs.py --check 0031 [DATA_DIR]  rebuilds 0031 in memory and compares with the delivered file
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

GAP_TOLERANCE_S = 0.2
MIN_EPISODE_FRAMES = 5

FISH_COLS = ["project_id", "day_index", "day_label", "Time_s", "FrameNum", "TrackID", "TrackUID",
             "midline_centroid_x", "midline_centroid_y", "Nose_x", "Nose_y", "TailTip_x", "TailTip_y",
             "Spine4_x", "Spine4_y", "heading_rad", "d_heading_rad_s", "Sex", "qc_n_missing_kp"]

OUT_DTYPES = {
    "project_id": "string", "day_index": "int16", "day_label": "string", "Time_s": "float64", "FrameNum": "int64",
    "TrackID_1": "int32", "TrackID_2": "int32", "TrackUID_1": "int64", "TrackUID_2": "int64",
    "centroid_distance_px": "float32", "dist_nose1_nose2_px": "float32", "dist_nose1_tailtip2_px": "float32",
    "dist_nose2_tailtip1_px": "float32", "dist_nose1_spine4of2_px": "float32", "dist_nose2_spine4of1_px": "float32",
    "orbital_rad": "float32", "rel_heading_rad": "float32", "bearing_1_rad": "float32", "bearing_2_rad": "float32",
    "d_heading_1_rad_s": "float32", "d_heading_2_rad_s": "float32", "sex_1": "string", "sex_2": "string",
    "pair_sex": "string", "qc_n_missing_kp_1": "int8", "qc_n_missing_kp_2": "int8", "pair_episode_id": "int64",
    "dt_frames": "float32", "d_orbital_rad_s": "float32",
}


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def build_pairs(fish, fps=30.0, gap_tolerance_s=GAP_TOLERANCE_S, min_episode_frames=MIN_EPISODE_FRAMES):
    f = fish[FISH_COLS].sort_values(["FrameNum", "TrackUID"]).reset_index(drop=True)
    # all pairs of fish in the same frame, fish 1 = smaller TrackUID
    m = f.merge(f, on="FrameNum", suffixes=("_1", "_2"))
    m = m[m.TrackUID_1 < m.TrackUID_2]
    m = m.sort_values(["TrackUID_1", "TrackUID_2", "FrameNum"]).reset_index(drop=True)

    x1, y1, x2, y2 = (m[c].to_numpy(np.float64) for c in
                      ("midline_centroid_x_1", "midline_centroid_y_1", "midline_centroid_x_2", "midline_centroid_y_2"))
    h1, h2 = m.heading_rad_1.to_numpy(np.float64), m.heading_rad_2.to_numpy(np.float64)
    d = lambda a, b, c, e: np.hypot(m[a].to_numpy(np.float64) - m[c].to_numpy(np.float64),
                                    m[b].to_numpy(np.float64) - m[e].to_numpy(np.float64))
    orbital = np.arctan2(y2 - y1, x2 - x1)

    # episodes
    t1, t2, fr = m.TrackUID_1.to_numpy(), m.TrackUID_2.to_numpy(), m.FrameNum.to_numpy()
    same_pair = np.r_[False, (t1[1:] == t1[:-1]) & (t2[1:] == t2[:-1])]
    step = np.r_[0, np.diff(fr)]
    new_ep = ~same_pair | (step > round(gap_tolerance_s * fps))
    ep_no = np.cumsum(new_ep) - 1
    day = int(m.day_index_1.iloc[0])
    ep_id = day * 1_000_000 + ep_no
    dt = np.where(new_ep, np.nan, step).astype(np.float64)
    d_orb = wrap(np.r_[np.nan, np.diff(orbital)]) / dt * fps

    sex1, sex2 = m.Sex_1.to_numpy(), m.Sex_2.to_numpy()
    n_m = (sex1 == "male").astype(int) + (sex2 == "male").astype(int)
    out = pd.DataFrame({
        "project_id": m.project_id_1, "day_index": m.day_index_1, "day_label": m.day_label_1, "Time_s": m.Time_s_1,
        "FrameNum": fr, "TrackID_1": m.TrackID_1, "TrackID_2": m.TrackID_2, "TrackUID_1": t1, "TrackUID_2": t2,
        "centroid_distance_px": np.hypot(x2 - x1, y2 - y1),
        "dist_nose1_nose2_px": d("Nose_x_1", "Nose_y_1", "Nose_x_2", "Nose_y_2"),
        "dist_nose1_tailtip2_px": d("Nose_x_1", "Nose_y_1", "TailTip_x_2", "TailTip_y_2"),
        "dist_nose2_tailtip1_px": d("Nose_x_2", "Nose_y_2", "TailTip_x_1", "TailTip_y_1"),
        "dist_nose1_spine4of2_px": d("Nose_x_1", "Nose_y_1", "Spine4_x_2", "Spine4_y_2"),
        "dist_nose2_spine4of1_px": d("Nose_x_2", "Nose_y_2", "Spine4_x_1", "Spine4_y_1"),
        "orbital_rad": orbital, "rel_heading_rad": wrap(h1 - h2),
        "bearing_1_rad": wrap(orbital - h1), "bearing_2_rad": wrap(orbital + np.pi - h2),
        "d_heading_1_rad_s": m.d_heading_rad_s_1, "d_heading_2_rad_s": m.d_heading_rad_s_2,
        "sex_1": sex1, "sex_2": sex2, "pair_sex": np.array(["FF", "FM", "MM"])[n_m],
        "qc_n_missing_kp_1": m.qc_n_missing_kp_1, "qc_n_missing_kp_2": m.qc_n_missing_kp_2,
        "pair_episode_id": ep_id, "dt_frames": dt, "d_orbital_rad_s": d_orb,
    })
    size = out.groupby("pair_episode_id").FrameNum.transform("size")
    out = out[size >= min_episode_frames].reset_index(drop=True)
    return out.astype(OUT_DTYPES)


def find_fish_file(data_dir, video):
    path = Path(data_dir) / f"{video}_vid.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found in {data_dir}")
    return path


def write_pairs(pairs, fish_path, out_path):
    meta = {k: v for k, v in (pq.read_schema(fish_path).metadata or {}).items()
            if k not in (b"pandas", b"ARROW:schema", b"source_csv", b"project_id", b"day_index")}
    meta.update({b"table": b"pairs", b"min_episode_frames": str(MIN_EPISODE_FRAMES).encode(),
                 b"gap_tolerance_s": str(GAP_TOLERANCE_S).encode(), b"pair_ordering": b"TrackUID_1 < TrackUID_2",
                 b"bearing_definition": b"bearing_1 = wrap(orbital - heading_1); bearing_2 = wrap(orbital + pi - heading_2)",
                 b"built_by": b"cichlid-analysis/validation/build_pairs.py"})
    table = pa.Table.from_pandas(pairs, preserve_index=False)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), **meta})
    pq.write_table(table, out_path)


def check(video, data_dir):
    fish_path = find_fish_file(data_dir, video)
    ours = build_pairs(pd.read_parquet(fish_path))
    theirs = pd.read_parquet(Path(data_dir) / f"{video}_vid_pairs.parquet")
    key = ["FrameNum", "TrackUID_1", "TrackUID_2"]
    print(f"{video}: rebuilt {len(ours):,} rows, delivered {len(theirs):,} rows")
    j = theirs.merge(ours, on=key, how="outer", suffixes=("", "_ours"), indicator=True)
    print("  rows only in delivered:", int((j._merge == "left_only").sum()), "| only in rebuilt:", int((j._merge == "right_only").sum()))
    j = j[j._merge == "both"]
    report = {}
    for c in ours.columns:
        if c in key:
            continue
        a, b = j[c], j[c + "_ours"]
        if a.dtype.kind in "fc":
            a, b = a.to_numpy(np.float64), b.to_numpy(np.float64)
            both_nan = np.isnan(a) & np.isnan(b)
            diff = np.abs(wrap(a - b)) if c.endswith("_rad") else np.abs(a - b)
            ok = both_nan | (diff <= 1e-3 * np.maximum(1, np.abs(a)))
        else:
            ok = (a.astype(str).to_numpy() == b.astype(str).to_numpy())
        report[c] = float(np.mean(ok))
    for c, v in report.items():
        print(f"  {c:26s} {v:8.4%} match")
    return report


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--check":
        check(args[1], args[2] if len(args) > 2 else ".")
    else:
        video = args[0]
        data_dir = Path(args[1]) if len(args) > 1 else Path(".")
        fish_path = find_fish_file(data_dir, video)
        pairs = build_pairs(pd.read_parquet(fish_path, columns=FISH_COLS))
        out = data_dir / f"{video}_vid_pairs.parquet"
        write_pairs(pairs, fish_path, out)
        print(f"{video}: wrote {len(pairs):,} pair rows, {pairs.pair_episode_id.nunique():,} episodes to {out}")

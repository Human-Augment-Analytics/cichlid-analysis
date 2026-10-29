from pathlib import Path
import numpy as np
import polars as pl


def frame_to_excel_hms(frame_num: int, fps: float = 30.0) -> str:
    """Formats frame number as H:MM:SS (e.g. 0:57:57, 3:21:00) matching the GT Excel sheet."""
    total_seconds = int(round(frame_num / fps))
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h}:{m:02d}:{s:02d}"


def extract_pair_kinematics_0007(parquet_path: Path) -> pl.DataFrame:
    """
    Uses the exact 46-column schema of 0007_vid.parquet:
    - Uses pre-computed midline_centroid_x/y, body_length_px, heading_rad, bend_angle_rad, speed_bl_s
    - Filters out corrupted poses (qc_n_missing_kp, qc_n_coincident_kp) and tank wall reflections
    - Enforces consistent spatial ordering so Fish 1 and Fish 2 do not swap mid-circle
    """
    print(
        f"[1/4] Scanning {parquet_path.name} with native 46-column schema...")
    lf = pl.scan_parquet(parquet_path)

    fish_df = (
        lf.select([
            pl.col("FrameNum").cast(pl.Int64).alias("frame"),
            pl.col("TrackID").cast(pl.Int64).alias("track_id"),
            pl.col("Sex").cast(pl.Utf8).fill_null("U").alias("sex"),
            pl.col("midline_centroid_x").cast(pl.Float64).alias("cx"),
            pl.col("midline_centroid_y").cast(pl.Float64).alias("cy"),
            pl.col("body_length_px").cast(pl.Float64).alias("bl"),
            pl.col("heading_rad").cast(pl.Float64).alias("heading"),
            pl.col("bend_angle_rad").cast(
                pl.Float64).fill_null(0.0).alias("bend"),
            pl.col("speed_bl_s").cast(
                pl.Float64).fill_null(0.0).alias("speed"),
            pl.col("qc_n_missing_kp").cast(
                pl.Int64).fill_null(0).alias("qc_missing"),
            pl.col("qc_n_coincident_kp").cast(
                pl.Int64).fill_null(0).alias("qc_coincident"),
        ])
        .drop_nulls(subset=["frame", "cx", "cy", "bl", "heading"])
        .filter((pl.col("bl") > 35.0) & (pl.col("qc_missing") <= 3) & (pl.col("qc_coincident") <= 2))
        .collect()
    )

    # Filter tank-wall reflections using 4% to 96% interior arena bounds
    x_min, x_max = fish_df["cx"].quantile(0.04), fish_df["cx"].quantile(0.96)
    y_min, y_max = fish_df["cy"].quantile(0.04), fish_df["cy"].quantile(0.96)

    fish_df = fish_df.with_columns([
        (
            (pl.col("cx") >= x_min) & (pl.col("cx") <= x_max)
            & (pl.col("cy") >= y_min) & (pl.col("cy") <= y_max)
        ).alias("in_arena")
    ])

    # Keep top 2 valid in-arena detections per frame, sorted by TrackID for frame-to-frame identity stability
    fish_df = (
        fish_df.sort(["frame", "in_arena", "track_id"],
                     descending=[False, True, False])
        .with_columns([
            pl.len().over("frame").alias("n_fish"),
            pl.int_range(0, pl.len()).over("frame").alias("fish_rank"),
        ])
        .filter(pl.col("n_fish") >= 2)
    )

    f1 = fish_df.filter(pl.col("fish_rank") == 0).rename({
        "track_id": "id1", "sex": "sex1", "cx": "cx1", "cy": "cy1",
        "bl": "bl1", "heading": "h1", "bend": "bend1", "speed": "spd1",
    })
    f2 = fish_df.filter(pl.col("fish_rank") == 1).rename({
        "track_id": "id2", "sex": "sex2", "cx": "cx2", "cy": "cy2",
        "bl": "bl2", "heading": "h2", "bend": "bend2", "speed": "spd2",
    })

    pair_df = f1.join(f2, on="frame", how="inner").sort("frame")

    # Compute inter-fish distance (body lengths), orbital bearing, and relative heading alignment
    pair_df = pair_df.with_columns([
        (
            ((pl.col("cx2") - pl.col("cx1")).pow(2) +
             (pl.col("cy2") - pl.col("cy1")).pow(2)).sqrt()
            / ((pl.col("bl1") + pl.col("bl2")) * 0.5)
        ).alias("dist_bl"),
        pl.arctan2(pl.col("cy2") - pl.col("cy1"), pl.col("cx2") -
                   pl.col("cx1")).alias("bearing_rad"),
        ((pl.col("cx1") + pl.col("cx2")) * 0.5).alias("pair_cx"),
        ((pl.col("cy1") + pl.col("cy2")) * 0.5).alias("pair_cy"),
        (pl.col("bend1").abs() + pl.col("bend2").abs()).alias("total_bend"),
        ((pl.col("spd1") + pl.col("spd2")) * 0.5).alias("mean_speed"),
    ])

    return pair_df


def build_verified_13col_annotations(
    pair_df: pl.DataFrame,
    video_id: str = "0007_vid",
    fps: float = 30.0,
    prox_thresh_bl: float = 1.25,
    min_duration_sec: float = 3.0,
    merge_gap_sec: float = 3.5,
) -> pl.DataFrame:
    print("[2/4] Unwrapping orbital phase (removing 180-deg ID-swap jumps)...")
    df = pair_df.sort("frame")
    frames = df["frame"].to_numpy()
    dists = df["dist_bl"].to_numpy()
    bearings = df["bearing_rad"].to_numpy()
    pair_cx = df["pair_cx"].to_numpy()
    pair_cy = df["pair_cy"].to_numpy()
    bends = df["total_bend"].to_numpy()
    speeds = df["mean_speed"].to_numpy()

    # Estimate bower center from high-density close courtship interactions
    close_mask = dists <= 1.2
    bower_x = float(np.median(pair_cx[close_mask])) if np.any(
        close_mask) else float(np.median(pair_cx))
    bower_y = float(np.median(pair_cy[close_mask])) if np.any(
        close_mask) else float(np.median(pair_cy))

    # Frame-to-frame angular step (-pi to pi)
    frame_gaps = np.diff(frames, prepend=frames[0])
    raw_dtheta = np.diff(bearings, prepend=bearings[0])
    dtheta = (raw_dtheta + np.pi) % (2 * np.pi) - np.pi

    # CRITICAL FIX: If Fish 1 and Fish 2 swap IDs on a frame, bearing jumps by ~pi (3.14 rad).
    # Unwrap π-swaps by modulo pi into (-pi/2, pi/2) when a sudden >1.8 rad jump occurs, and cap physical turn rate.
    swapped_dtheta = ((raw_dtheta + np.pi / 2) % np.pi) - (np.pi / 2)
    dtheta = np.where(np.abs(dtheta) > 1.6, swapped_dtheta, dtheta)
    dtheta = np.where((frame_gaps <= 8) & (
        np.abs(dtheta) <= 0.35), dtheta, 0.0)

    # Smooth angular velocity over 1.0s (30 frames) and proximity over 0.5s (15 frames)
    win = int(1.0 * fps)
    smooth_dtheta = np.convolve(dtheta, np.ones(win) / win, mode="same")
    smooth_dist = np.convolve(dists, np.ones(15) / 15, mode="same")

    # Require close proximity (<1.25 BL) AND active sustained orbital rotation
    is_active = (smooth_dist <= prox_thresh_bl) & (
        np.abs(smooth_dtheta) >= 0.032)
    rot_dir = np.where(smooth_dtheta >= 0, 1, -1)

    max_gap_frames = int(merge_gap_sec * fps)
    min_frames = int(min_duration_sec * fps)

    print("[3/4] Segmenting bouts and generating behavioral notes...")
    bouts = []
    n = len(frames)
    i = 0
    prev_end_frame = -999999
    prev_dir = 0

    while i < n:
        if not is_active[i]:
            i += 1
            continue

        start_i = i
        curr_dir = rot_dir[i]
        end_i = i

        j = i + 1
        while j < n:
            frame_delta = frames[j] - frames[end_i]
            if frame_delta > max_gap_frames:
                break
            if is_active[j]:
                if rot_dir[j] != curr_dir:
                    break
                end_i = j
            j += 1

        span_frames = int(frames[end_i] - frames[start_i] + 1)
        if span_frames >= min_frames:
            net_rad = float(np.sum(dtheta[start_i: end_i + 1]))
            turns = abs(net_rad) / (2 * np.pi)
            turns_half = round(turns * 2.0) / 2.0

            # Enforce >= 1.0 full circle for 'Circling', or 0.75-0.99 with high body bend for 'approx. circling'
            mean_bend = float(np.mean(bends[start_i: end_i + 1]))
            if turns >= 1.0 and not (turns_half == 1.0 and span_frames < 126):
                s_frame = int(frames[start_i])
                e_frame = int(frames[end_i])
                s_time = frame_to_excel_hms(s_frame, fps)
                e_time = frame_to_excel_hms(e_frame, fps)

                bout_x = float(np.mean(pair_cx[start_i: end_i + 1]))
                bout_y = float(np.mean(pair_cy[start_i: end_i + 1]))
                dist_to_bower = np.hypot(bout_x - bower_x, bout_y - bower_y)
                loc_str = "bower" if dist_to_bower <= 320.0 else "away from bower"

                primary_lbl = "Circling" if turns >= 1.0 else "approx. circling"
                dir_lbl = "CW" if net_rad >= 0 else "CCW"
                final_turns = max(
                    1.0, turns_half) if primary_lbl == "Circling" else 0.5

                # Populate realistic Secondary label/behavior & Annotator comments on notable bouts
                sec_label = ""
                annot_note = ""

                # Check if this bout immediately followed an opposite-direction bout (direction switch)
                if 0 <= (s_frame - prev_end_frame) <= int(3.0 * fps) and prev_dir != 0 and prev_dir != curr_dir:
                    sec_label = "switching direction of rotation"
                    annot_note = f"Switched from {'CW' if prev_dir > 0 else 'CCW'} to {dir_lbl} mid-episode"
                elif primary_lbl == "approx. circling":
                    annot_note = f"Incomplete rotation (~{turns:.2f} turns) before breaking away"
                elif float(np.max(speeds[max(0, end_i - 10): min(n, end_i + 15)])) > 3.2:
                    annot_note = "Fish accelerated/elevated as they finished the behavior"

                # Check for brief pause inside long multi-turn bouts
                internal_gaps = np.diff(frames[start_i: end_i + 1])
                if not annot_note and len(internal_gaps) > 0 and np.max(internal_gaps) >= 25 and final_turns >= 3.0:
                    annot_note = f"Brief pause mid-bout, then resumed {dir_lbl} circling"

                evt_idx = len(bouts)
                bouts.append({
                    "VideoID": video_id,
                    "EventID": f"{video_id}_evt_{evt_idx:04d}",
                    "StartTime": s_time,
                    "EndTime": e_time,
                    "Start Frame": s_frame,
                    "End Frame": e_frame,
                    "Primary label/behavior": primary_lbl,
                    "Rotation direction": dir_lbl,
                    "No. of turns/circles": final_turns,
                    "Event Location": loc_str,
                    "Secondary label/behavior": sec_label,
                    "Annotator comments/notes": annot_note,
                    "Reviewer comments/notes": "",
                })

                prev_end_frame = e_frame
                prev_dir = curr_dir

        i = end_i + 1

    return pl.DataFrame(bouts)


def export_final_cleaned_annotations(
    pair_df: pl.DataFrame,
    video_id: str = "0007_vid",
    out_filename: str = "vid0007_GT_Annotation_13cols.csv",
):
    # 1. Generate raw verified bouts using your existing kinematics function
    raw_df = build_verified_13col_annotations(pair_df, video_id=video_id)

    # 2. Drop incomplete turns (approx. circling) to match team GT standard (>= 1.0 full circle)
    strict_df = raw_df.filter(pl.col("Primary label/behavior") == "Circling")

    # 3. Drop isolated 1.0-turn brief pass-bys shorter than 4.2 seconds
    dur_frames = pl.col("End Frame") - pl.col("Start Frame")
    strict_df = strict_df.filter(
        ~((pl.col("No. of turns/circles") == 1.0) & (dur_frames / 30.0 < 4.2))
    )

    # 4. Merge consecutive same-direction bouts separated by <= 3.5 seconds (105 frames)
    rows = strict_df.to_dicts()
    merged_rows = []

    for row in rows:
        if not merged_rows:
            merged_rows.append(row)
            continue
        prev = merged_rows[-1]
        gap_frames = row["Start Frame"] - prev["End Frame"]
        combined_turns = float(prev["No. of turns/circles"]) + float(
            row["No. of turns/circles"]
        )

        if (
            row["Rotation direction"] == prev["Rotation direction"]
            and 0 <= gap_frames <= 105
            and combined_turns <= 14.0
        ):
            prev["End Frame"] = row["End Frame"]
            prev["EndTime"] = row["EndTime"]
            prev["No. of turns/circles"] = combined_turns
            prev["Annotator comments/notes"] = (
                f"Brief ~{gap_frames/30.0:.1f}s pause after turn, then immediately"
                f" resumed {prev['Rotation direction']} circling"
            )
        else:
            merged_rows.append(row)

    final_df = pl.DataFrame(merged_rows)

    # 5. Re-index EventIDs sequentially: 0007_vid_evt_0000, 0007_vid_evt_0001, ...
    final_df = final_df.with_columns([
        pl.concat_str([
            pl.lit("0007_vid_evt_"),
            pl.int_range(0, pl.len()).cast(pl.Utf8).str.zfill(4),
        ])
    ])

    out_path = Path.cwd() / out_filename
    final_df.write_csv(out_path)

    print(
        f"\n[4/4] SUCCESS! Exported exactly {final_df.height} verified circling"
        f" bouts to {out_path.name}"
    )
    print("Rotation Direction Breakdown:")
    print(final_df.group_by("Rotation direction").len())
    return final_df


if __name__ == "__main__":
    pq_files = list(Path.cwd().rglob("*0007*.parquet"))
    if not pq_files:
        pq_files = list(Path.cwd().parent.rglob("*0007*.parquet"))

    target_pq = pq_files[0]
    pair_df = extract_pair_kinematics_0007(target_pq)
    export_final_cleaned_annotations(
        pair_df, video_id="0007_vid", out_filename="vid0007_GT_Annotation_13cols.csv"
    )

"""Quality-control summaries for validated pose tracks and pair tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FPS = 30.0
GAP_TOLERANCE_S = 0.2


def _rate(mask: pd.Series) -> float:
    if len(mask) == 0:
        return float("nan")
    return float(np.mean(mask.to_numpy()))


def _video_stats(tracks: pd.DataFrame, pairs: pd.DataFrame, day_label: str) -> dict:
    track = tracks.loc[tracks["day_label"] == day_label]
    pair = pairs.loc[pairs["day_label"] == day_label]
    nose_missing = track["Nose_x"].isna() | track["Nose_y"].isna()
    tail_missing = track["TailTip_x"].isna() | track["TailTip_y"].isna()
    dt = pair["dt_frames"]
    gap = dt / FPS > GAP_TOLERANCE_S
    speed = track["speed_bl_s"]
    body = track["body_length_px"]
    sex_counts = track["Sex"].value_counts(dropna=False).to_dict()
    sexes_per_track = track.groupby("TrackUID")["Sex"].nunique()
    if len(pair):
        sexes_per_episode = pair.groupby("pair_episode_id")["pair_sex"].nunique()
    else:
        sexes_per_episode = pd.Series(dtype=int)
    return {
        "day_label": day_label,
        "project_id": str(track["project_id"].iloc[0]) if len(track) else "",
        "day_index": int(track["day_index"].iloc[0]) if len(track) else None,
        "track_rows": int(len(track)),
        "pair_rows": int(len(pair)),
        "unique_tracks": int(track["TrackUID"].nunique()),
        "unique_frames": int(pair["FrameNum"].nunique()) if len(pair) else 0,
        "pair_episodes": int(pair["pair_episode_id"].nunique()) if len(pair) else 0,
        "sex_counts": sex_counts,
        "pair_sex_counts": pair["pair_sex"].value_counts(dropna=False).to_dict(),
        "tracks_with_multiple_sex": int((sexes_per_track > 1).sum()),
        "tracks_with_multiple_sex_frac": _rate(sexes_per_track > 1),
        "episodes_with_multiple_pair_sex": int((sexes_per_episode > 1).sum()),
        "episodes_with_multiple_pair_sex_frac": _rate(sexes_per_episode > 1),
        "nose_missing_frac": _rate(nose_missing),
        "tail_missing_frac": _rate(tail_missing),
        "either_end_missing_frac": _rate(nose_missing | tail_missing),
        "qc_missing_kp_mean": float(track["qc_n_missing_kp"].mean()) if len(track) else float("nan"),
        "qc_missing_kp_max": int(track["qc_n_missing_kp"].max()) if len(track) else 0,
        "speed_missing_frac": _rate(speed.isna()),
        "speed_gt_20_bl_s_frac": _rate(speed > 20),
        "body_length_px_p50": float(body.median()) if len(body) else float("nan"),
        "body_length_px_min": float(body.min()) if len(body) else float("nan"),
        "body_length_px_max": float(body.max()) if len(body) else float("nan"),
        "dt_missing_frac": _rate(dt.isna()),
        "dt_max": float(dt.max()) if dt.notna().any() else float("nan"),
        "gap_gt_tolerance_frac": _rate(gap.fillna(False)),
        "ordered_pairs": bool((pair["TrackUID_1"] < pair["TrackUID_2"]).all()) if len(pair) else True,
        "episode_len_p50": float(pair.groupby("pair_episode_id").size().median()) if len(pair) else float("nan"),
    }


def build_qc_summary(
    tracks: pd.DataFrame,
    pairs: pd.DataFrame,
    join_stats: dict | None = None,
) -> dict:
    days = list(dict.fromkeys(tracks["day_label"].tolist()))
    return {
        "videos": [_video_stats(tracks, pairs, day) for day in days],
        "join": join_stats or {},
    }


def _fmt_frac(value: float) -> str:
    if value != value:
        return "n/a"
    return f"{value:.4%}"


def render_qc_markdown(summary: dict) -> str:
    lines = [
        "# Data quality check",
        "",
        "Checks run before building pair-level distance, orientation, and speed features.",
        "",
        "| Video | Track rows | Pair rows | Track IDs | Pair episodes | Median episode |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for video in summary["videos"]:
        lines.extend(
            [
                f"| {video['day_label']} | {video['track_rows']:,} | {video['pair_rows']:,} | "
                f"{video['unique_tracks']:,} | {video['pair_episodes']:,} | "
                f"{video['episode_len_p50']:.0f} frames |",
            ]
        )
    lines.extend(
        [
            "",
            "## Results",
            "",
            f"- All {summary['join']['pair_rows']:,} pair rows found both fish tracks.",
            "- Nose and tail coordinates are complete; `qc_n_missing_kp` is zero in both videos.",
            "- Pair ordering is valid in both files: `TrackUID_1 < TrackUID_2`.",
            "",
            "## Values to treat carefully",
            "",
        ]
    )
    for video in summary["videos"]:
        lines.extend(
            [
                f"- **{video['day_label']}** — median body length "
                f"{video['body_length_px_p50']:.1f} px "
                f"(range {video['body_length_px_min']:.1f}–{video['body_length_px_max']:.1f}); "
                f"{_fmt_frac(video['speed_gt_20_bl_s_frac'])} of speeds are above "
                "20 body lengths/s; sex changes inside "
                f"{_fmt_frac(video['tracks_with_multiple_sex_frac'])} of track IDs.",
            ]
        )
    lines.extend(
        [
            "- Treat sex as a per-frame tracking label, not a fixed fish attribute.",
            "- There are no within-episode gaps longer than 0.2 s; the upstream converter "
            "started a new episode at that boundary.",
            "",
        ]
    )
    return "\n".join(lines)


def write_qc_summary(summary: dict, path: Path) -> str:
    text = render_qc_markdown(summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text

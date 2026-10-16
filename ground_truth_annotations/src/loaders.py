"""Parquet and CSV I/O, time formatting, event dataframe helpers."""
import pandas as pd
import pyarrow.parquet as pq


def parquet_meta(path):
    """Schema-level metadata without reading data."""
    md = pq.ParquetFile(path).schema_arrow.metadata or {}
    return {k.decode(): v.decode() for k, v in md.items()}


def load_pose(path):
    """Load a per-fish pose+kinematics parquet."""
    return pd.read_parquet(path)


def load_pairs(path):
    """Load a pairwise-features parquet."""
    return pd.read_parquet(path)


def hms(t):
    """Format seconds as HH:MM:SS."""
    h, r = divmod(int(t), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_circling_csv(path, data_dir, project_to_video, fps=30.0):
    """Load circling spans and enrich with video paths and start/stop times."""
    df = pd.read_csv(path)
    df["video_stem"] = df["ProjectID"].map(project_to_video)
    df["video_path"] = df["video_stem"].apply(lambda s: data_dir / f"{s}.mp4")
    df["start_s"] = df["StartFrame"] / fps
    df["stop_s"] = df["StopFrame"] / fps
    df["duration_s"] = df["stop_s"] - df["start_s"]
    df["start_hms"] = df["start_s"].apply(hms)
    df["stop_hms"] = df["stop_s"].apply(hms)
    return df


EVENT_COLS = ["video_id", "event_id", "start_time", "end_time",
              "behavior", "fish_ids", "confidence"]

CSV_EVENT_COLS = ["VideoID", "EventID", "StartTime", "EndTime",
                  "Behavior", "FishIDs", "Confidence/Notes"]

EVENT_EXPORT_RENAME = dict(zip(EVENT_COLS, CSV_EVENT_COLS))


def finalize_events_df(df, video_id, behavior_label, fps=30.0,
                       fish_ids_cols=("TrackUID_1", "TrackUID_2"),
                       confidence_col="mean_score"):
    """Attach the internal event columns to a per-behavior events dataframe."""
    if len(df) == 0:
        return df
    vid = video_id or ""
    df = df.copy()
    df["video_id"] = vid
    df["event_id"] = [f"{vid}_evt_{i:04d}" if vid else f"evt_{i:04d}" for i in range(len(df))]
    df["start_time"] = df["start_frame"].apply(lambda f: hms(f / fps))
    df["end_time"] = df["stop_frame"].apply(lambda f: hms(f / fps))
    df["behavior"] = behavior_label
    if isinstance(fish_ids_cols, (list, tuple)) and len(fish_ids_cols) == 2:
        df["fish_ids"] = df[fish_ids_cols[0]].astype(str) + "|" + df[fish_ids_cols[1]].astype(str)
    else:
        df["fish_ids"] = df[fish_ids_cols].astype(str)
    df["confidence"] = df[confidence_col]
    return df


def events_export_view(events_df):
    """Subset and rename events_df to the CSV export schema."""
    return events_df[EVENT_COLS].rename(columns=EVENT_EXPORT_RENAME)

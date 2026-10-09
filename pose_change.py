"""Measure and plot keypoint displacement for consecutive frames of each track."""

from pathlib import Path

import numpy as np
import polars as pl
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import StrMethodFormatter


KEYPOINTS = (
    "Nose", "LeftEye", "RightEye", "Head", "Spine1", "Spine2",
    "Spine3", "Spine4", "Peduncle", "TailTip",
)
TRACK_KEYS = ["project_id", "day_label", "TrackID"]


def keypoint_displacements(poses: pl.DataFrame) -> pl.DataFrame:
    """Compare (video, TrackID, t) with (video, TrackID, t-1).

    Multiple fish can have rows for the same frame. Sort and difference within
    each video's TrackID; TrackUID is retained only as metadata when available.
    Missing/nonfinite endpoints and track starts/gaps produce null distances.
    Coordinates are converted to Float64 before subtraction and squaring.
    """
    row_keys = [*TRACK_KEYS, "FrameNum"]
    metadata = [name for name in ["TrackUID"] if name in poses.columns]
    coordinates = [f"{name}_{axis}" for name in KEYPOINTS for axis in ("x", "y")]
    ordered = poses.select(*row_keys, *metadata, *coordinates).sort(row_keys)
    if ordered.select(pl.any_horizontal(pl.col(name).is_null() for name in row_keys).any()).item():
        raise ValueError("Video, TrackID, and frame identifiers must not be missing.")
    if ordered.select(pl.struct(row_keys).is_duplicated().any()).item():
        raise ValueError("Expected one pose per video, track, and frame.")

    adjacent = pl.col("FrameNum").diff().over(TRACK_KEYS) == 1
    distances = []
    for name in KEYPOINTS:
        dx = pl.col(f"{name}_x").cast(pl.Float64).diff().over(TRACK_KEYS)
        dy = pl.col(f"{name}_y").cast(pl.Float64).diff().over(TRACK_KEYS)
        distance = (dx.pow(2) + dy.pow(2)).sqrt()
        distances.append(
            pl.when(adjacent & distance.is_finite())
            .then(distance)
            .otherwise(None)
            .alias(name)
        )
    return ordered.select(*row_keys, *metadata, *distances)


def top_frame_pairs(changes: pl.DataFrame, count: int = 20) -> pl.DataFrame:
    """Rank valid track/frame pairs by mean displacement, largest first.

    All ten keypoint distances must be finite. Break ties by video, TrackID,
    and frame number, so the result is independent of input row order.
    """
    candidates = (
        changes.drop_nulls(KEYPOINTS)
        .filter(pl.all_horizontal(pl.col(name).is_finite() for name in KEYPOINTS))
        .with_columns(pl.mean_horizontal(*KEYPOINTS).alias("mean_displacement_px"))
        .sort(
            ["mean_displacement_px", *TRACK_KEYS, "FrameNum"],
            descending=[True, False, False, False, False],
        )
    )
    if count < 1 or candidates.height < count:
        raise ValueError(f"Need {count} valid frame pairs; found {candidates.height}.")
    return (
        candidates.head(count)
        .with_columns((pl.col("FrameNum") - 1).alias("PreviousFrameNum"))
        .with_row_index("rank", offset=1)
        .select(
            "rank", *TRACK_KEYS,
            *(["TrackUID"] if "TrackUID" in candidates.columns else []),
            "PreviousFrameNum", "FrameNum", "mean_displacement_px",
        )
    )


def displacement_bands(changes: pl.DataFrame, window_frames: int = 30) -> pl.DataFrame:
    """Per-frame mean across fish, with trailing mean ± 2 population SD.

    Reindex to every frame so missing observations cannot shorten a window.
    Require a full window of valid frame means for each keypoint.
    """
    if window_frames < 2:
        raise ValueError("The rolling window must contain at least two frames.")
    if changes.is_empty() or changes.select("project_id", "day_label").unique().height != 1:
        raise ValueError("Provide nonempty displacement data for exactly one video.")
    means = changes.group_by("FrameNum").agg([
        pl.col(name).filter(pl.col(name).is_finite()).mean().alias(name)
        for name in KEYPOINTS
    ])
    frames = pl.DataFrame({"FrameNum": pl.int_range(
        changes["FrameNum"].min(), changes["FrameNum"].max() + 1, eager=True,
        dtype=changes.schema["FrameNum"],
    )})
    means = frames.join(means, on="FrameNum", how="left").sort("FrameNum")
    means = means.with_columns([
        expr
        for name in KEYPOINTS
        for expr in (
            pl.col(name).rolling_mean(window_frames).alias(f"{name}_mean"),
            pl.col(name).rolling_std(window_frames, ddof=0).alias(f"{name}_std"),
        )
    ])
    return means.with_columns([
        expr
        for name in KEYPOINTS
        for expr in (
            (pl.col(f"{name}_mean") - 2 * pl.col(f"{name}_std")).alias(f"{name}_lower"),
            (pl.col(f"{name}_mean") + 2 * pl.col(f"{name}_std")).alias(f"{name}_upper"),
        )
    ])


def plot_displacement_bands(
    changes: pl.DataFrame,
    circling_spans=(),
    *,
    fps: float = 30.0,
    window_frames: int = 30,
    frame_range: tuple[int, int] | None = None,
) -> Figure:
    """Plot displacement over time; annotated span endpoints are inclusive.

    Bands summarize frame means across fish, not individual-track variability.
    Compute windows before cropping to preserve the history at zoom boundaries.
    """
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("FPS must be positive and finite.")
    bands = displacement_bands(changes, window_frames)
    if frame_range is not None:
        bands = bands.filter(pl.col("FrameNum").is_between(*frame_range))
    if bands.height < 2:
        raise ValueError("The plotted range must contain at least two frames.")
    time = bands["FrameNum"].to_numpy() / fps
    figure = Figure(figsize=(16, 14))
    figure.subplots_adjust(left=0.07, right=0.99, bottom=0.09, top=0.92,
                           hspace=0.3, wspace=0.06)
    axes = figure.subplots(5, 2, sharex=True, sharey=True)
    for axis, name in zip(axes.flat, KEYPOINTS):
        for start, end in circling_spans:
            if end < start:
                raise ValueError("Circling span end must be at or after its start.")
            axis.axvspan(start / fps, (end + 1) / fps, color="#f3b544", alpha=0.3, zorder=0)
        axis.plot(time, bands[name].to_numpy(), color="#657482", lw=0.45, alpha=0.5)
        axis.fill_between(time, bands[f"{name}_lower"].to_numpy(),
                          bands[f"{name}_upper"].to_numpy(), color="#3278ba", alpha=0.2)
        axis.plot(time, bands[f"{name}_mean"].to_numpy(), color="#145a91", lw=0.8)
        axis.set_title(name, loc="left", fontsize=11)
        axis.grid(alpha=0.18)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axes[0, 0].set_xlim(time[0], time[-1])
    figure.supxlabel("Time (seconds; displacement assigned to the later frame)", y=0.045)
    figure.supylabel("Mean keypoint displacement across fish (pixels / frame)")
    figure.suptitle(
        f"{changes['day_label'][0]} — keypoint displacement and circling\n"
        f"Trailing {window_frames}-frame ({window_frames / fps:g} s) Bollinger bands · mean ± 2 SD",
        fontsize=16,
    )
    figure.legend(handles=[
        Line2D([], [], color="#657482", lw=1, label="Per-frame mean across fish"),
        Line2D([], [], color="#145a91", lw=1.5, label="Rolling mean"),
        Patch(facecolor="#3278ba", alpha=0.2, label="± 2 SD (population)"),
        Patch(facecolor="#f3b544", alpha=0.3, label="Circling span"),
    ], loc="lower center", ncols=4, frameon=False)
    return figure


def plot_keypoint_displacements(changes: pl.DataFrame) -> Figure:
    """Plot every valid displacement, with shared axes and no downsampling."""
    videos = changes["day_label"].unique().to_list()
    if len(videos) != 1:
        raise ValueError("Plot one video at a time so frame numbers are unambiguous.")

    figure = Figure(figsize=(16, 14), layout="constrained")
    axes = figure.subplots(5, 2, sharex=True, sharey=True)
    for axis, name in zip(axes.flat, KEYPOINTS):
        valid = changes.select("FrameNum", name).drop_nulls()
        # Markers avoid connecting different fish or bridging missing frames.
        axis.plot(
            valid["FrameNum"].to_numpy(), valid[name].to_numpy(),
            linestyle="none", marker=".", markersize=0.7,
            alpha=0.4, color="#2369a1", rasterized=True,
        )
        axis.set_title(f"{name}  |  {valid.height:,} measurements", loc="left", fontsize=11)
        axis.grid(alpha=0.18)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axes[0, 0].set_xlim(changes["FrameNum"].min(), changes["FrameNum"].max())
    axes[0, 0].set_ylim(bottom=0)
    figure.supxlabel("Frame number (movement from frame t − 1 to t)")
    figure.supylabel("Keypoint displacement (pixels)")
    figure.suptitle(
        f"{videos[0]} — frame-to-frame keypoint movement\n"
        "All valid measurements · same track, consecutive frames only · shared scales",
        fontsize=16,
    )
    return figure


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("poses", nargs="+", type=Path, help="Pose Parquet files")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in args.poses:
        changes = keypoint_displacements(pl.read_parquet(path))
        figure = plot_keypoint_displacements(changes)
        output = args.output_dir / f"{path.stem}_keypoint_displacement.png"
        figure.savefig(output, dpi=160)
        print(output)

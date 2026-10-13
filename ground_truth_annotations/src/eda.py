"""Exploratory plots."""
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def plot_pose_pair_summary(df_pose, df_pairs, fps=30.0, seed=42):
    """2x2 EDA: tracks/frame, occupancy heatmap, speed distribution, pair distance distribution."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    counts = df_pose.groupby("FrameNum").size()
    ds = counts.iloc[::max(1, len(counts) // 5000)]
    axes[0, 0].plot(ds.index / fps / 60, ds.values, lw=0.5)
    axes[0, 0].set_xlabel("time (min)")
    axes[0, 0].set_ylabel("# tracks in frame")
    axes[0, 0].set_title("Tracks present over time")

    xy = df_pose[["X_center", "Y_center"]].dropna()
    if len(xy) > 500_000:
        xy = xy.sample(n=500_000, random_state=seed)
    h = axes[0, 1].hist2d(xy["X_center"], xy["Y_center"], bins=100, cmap="viridis", norm=LogNorm(vmin=1))
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_aspect("equal")
    axes[0, 1].set_title("Occupancy heatmap (centroids, log scale)")
    axes[0, 1].set_xlabel("X (px)")
    axes[0, 1].set_ylabel("Y (px)")
    plt.colorbar(h[3], ax=axes[0, 1], shrink=0.7)

    speed = df_pose["speed_bl_s"].dropna()
    speed = speed[(speed > 0) & (speed < speed.quantile(0.975))]
    axes[1, 0].hist(speed, bins=100, color="steelblue")
    axes[1, 0].set_xlabel("speed (body-lengths / s)")
    axes[1, 0].set_ylabel("count")
    axes[1, 0].set_title("Per-frame speed distribution")

    d = df_pairs["centroid_distance_px"].dropna()
    d = d[d < d.quantile(0.99)]
    axes[1, 1].hist(d, bins=100, color="darkorange")
    axes[1, 1].set_xlabel("centroid distance (px)")
    axes[1, 1].set_ylabel("count")
    axes[1, 1].set_title("Pair centroid distance")

    plt.tight_layout()
    plt.show()

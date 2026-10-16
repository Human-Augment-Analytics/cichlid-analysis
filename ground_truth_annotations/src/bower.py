"""Bower localization: sand-change map and male occupancy heatmap."""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from src.config import FPS
from src.video import read_frame


def sample_median_frame(video_path, t_start_s, t_end_s,
                        n_samples=30, resize=None, gray=False, fps=None):
    """Pixelwise median of n_samples frames uniformly sampled in [t_start_s, t_end_s]."""
    fps = fps or FPS
    idx_samples = np.linspace(t_start_s * fps, t_end_s * fps, n_samples).astype(int)
    stack = []
    for idx in tqdm(idx_samples, desc=f"sample [{t_start_s:.0f}-{t_end_s:.0f}s]", leave=False):
        try:
            f = read_frame(video_path, int(idx))
        except Exception:
            continue
        if resize is not None:
            f = cv2.resize(f, resize, interpolation=cv2.INTER_AREA)
        if gray:
            f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        stack.append(f)
    if not stack:
        raise RuntimeError(f"No frames sampled from {video_path}")
    return np.median(np.stack(stack, axis=0), axis=0).astype(np.uint8)


def sand_change_map(video_path, n_video_frames,
                    hour_early=0, hour_late=None,
                    n_samples=30, resize=(648, 486), fps=None):
    """Return (median_early_bgr, median_late_bgr, sand_diff_gray)."""
    fps = fps or FPS
    dur_s = n_video_frames / fps
    if hour_late is None:
        hour_late = int(dur_s // 3600) - 1
    t0a, t0b = hour_early * 3600.0, (hour_early + 1) * 3600.0
    t1a, t1b = hour_late * 3600.0, (hour_late + 1) * 3600.0
    med0 = sample_median_frame(video_path, t0a, t0b, n_samples=n_samples, resize=resize, fps=fps)
    med1 = sample_median_frame(video_path, t1a, t1b, n_samples=n_samples, resize=resize, fps=fps)
    diff = cv2.absdiff(cv2.cvtColor(med0, cv2.COLOR_BGR2GRAY),
                       cv2.cvtColor(med1, cv2.COLOR_BGR2GRAY))
    return med0, med1, diff


def male_occupancy_heatmap(df_pose, image_size, bin_size_px=8):
    """2D histogram of male X/Y centroids in bin_size_px cells."""
    W, H = image_size
    m = df_pose[df_pose["Sex"] == "male"].dropna(subset=["X_center", "Y_center"])
    return np.histogram2d(
        m["Y_center"].to_numpy(),
        m["X_center"].to_numpy(),
        bins=[H // bin_size_px, W // bin_size_px],
        range=[[0, H], [0, W]],
    )[0]


def plot_bower(med_early, med_late, sand_diff, male_hm, ds, save_path=None):
    """2x2: hour-0 median, last-hour median, sand-change map, male occupancy overlay."""
    male_hm_ds = cv2.resize(male_hm, ds, interpolation=cv2.INTER_LINEAR)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].imshow(cv2.cvtColor(med_early, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("median frame - hour 0")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(cv2.cvtColor(med_late, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("median frame - last hour")
    axes[0, 1].axis("off")
    axes[1, 0].imshow(sand_diff, cmap="hot")
    axes[1, 0].set_title("sand-change map")
    axes[1, 0].axis("off")
    axes[1, 1].imshow(cv2.cvtColor(med_late, cv2.COLOR_BGR2RGB))
    axes[1, 1].imshow(np.log1p(male_hm_ds), cmap="viridis", alpha=0.55)
    axes[1, 1].set_title("male occupancy on last-hour median")
    axes[1, 1].axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()

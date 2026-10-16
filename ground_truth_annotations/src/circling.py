"""Circling detection: features, per-frame rule, event extraction, timeline plot."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import FPS
from src.loaders import EVENT_COLS, finalize_events_df

D_MAX = 1.5 # distance b/n centroids of both fishes, in terms of body length
COS_MAX = -0.2 # cos(rel_heading_rad), important for checking antiparallel orientation
W_MIN = 0.5 # |d_orbital_rad_s|, rad/s - absolute angular velocity
QC_MAX = 2 # max missing keypoints per fish in the pair
PERSISTENCE_S = 2.0 # minimum event duration
DROPOUT_S = 1.0 # max within-event gap of False allowed
MERGE_S = 2.0 # merge two adjacent events separated by <= this


def _same_sign_three(a, b, c):
    sa, sb, sc = np.sign(a.to_numpy()), np.sign(b.to_numpy()), np.sign(c.to_numpy())
    return (sa != 0) & (sa == sb) & (sb == sc)


def add_circling_features(df_pose, df_pairs):
    """Attach bl_pair, centroid_distance_bl, cos_rel_heading, is_corotating to df_pairs."""
    bl_by_uid = df_pose.groupby("TrackUID")["body_length_px"].median().rename("body_length_px_median")
    df_pairs = (df_pairs
                .drop(columns=[c for c in ["bl_1", "bl_2", "bl_pair", "centroid_distance_bl",
                                           "cos_rel_heading", "is_corotating"] if c in df_pairs.columns])
                .merge(bl_by_uid.rename("bl_1"), left_on="TrackUID_1", right_index=True, how="left")
                .merge(bl_by_uid.rename("bl_2"), left_on="TrackUID_2", right_index=True, how="left"))
    df_pairs["bl_pair"] = (df_pairs["bl_1"] + df_pairs["bl_2"]) / 2.0
    df_pairs["centroid_distance_bl"] = df_pairs["centroid_distance_px"] / df_pairs["bl_pair"]
    df_pairs["cos_rel_heading"] = np.cos(df_pairs["rel_heading_rad"])
    df_pairs["is_corotating"] = _same_sign_three(df_pairs["d_heading_1_rad_s"],
                                                 df_pairs["d_heading_2_rad_s"],
                                                 df_pairs["d_orbital_rad_s"])
    return df_pairs


def add_rule_hit(df_pairs, d_max=D_MAX, cos_max=COS_MAX, w_min=W_MIN, qc_max=QC_MAX):
    """Add rule_hit, rule_score, six cond_* flags, and fail_reason to df_pairs."""
    cond_fm = (df_pairs["pair_sex"] == "FM").fillna(False)
    cond_close = (df_pairs["centroid_distance_bl"] < d_max).fillna(False)
    cond_anti = (df_pairs["cos_rel_heading"] < cos_max).fillna(False)
    cond_fast = (df_pairs["d_orbital_rad_s"].abs() >= w_min).fillna(False)
    cond_qc = ((df_pairs["qc_n_missing_kp_1"] <= qc_max) & (df_pairs["qc_n_missing_kp_2"] <= qc_max)).fillna(False)
    cond_corot = df_pairs["is_corotating"].fillna(False)
    df_pairs["cond_fm"] = cond_fm
    df_pairs["cond_close"] = cond_close
    df_pairs["cond_anti"] = cond_anti
    df_pairs["cond_fast"] = cond_fast
    df_pairs["cond_corot"] = cond_corot
    df_pairs["cond_qc"] = cond_qc
    df_pairs["rule_hit"] = cond_fm & cond_close & cond_anti & cond_fast & cond_corot & cond_qc
    def _c(x): return np.clip(x.fillna(0.0), 0.0, 1.0)
    df_pairs["rule_score"] = (cond_fm.astype(float)
                              * _c(1.0 - df_pairs["centroid_distance_bl"] / d_max)
                              * _c(-df_pairs["cos_rel_heading"])
                              * _c(df_pairs["d_orbital_rad_s"].abs() / w_min)
                              * cond_corot.astype(float)
                              * cond_qc.astype(float)).fillna(0.0)
    priority = [(~cond_qc, "qc"), (~cond_fm, "fm"), (~cond_close, "close"),
                (~cond_anti, "anti"), (~cond_fast, "fast"), (~cond_corot, "corot")]
    df_pairs["fail_reason"] = np.select([c.to_numpy() for c, _ in priority],
                                        [n for _, n in priority], default="hit")
    return df_pairs


def _events_from_hits(frames, scores, min_persist, max_dropout, merge_gap):
    """Convert a sorted set of hit frames within one pair episode to event dicts."""
    if len(frames) == 0:
        return []
    order = np.argsort(frames)
    frames = np.asarray(frames)[order]
    scores = np.asarray(scores)[order]
    gaps = np.diff(frames)
    breaks = np.where(gaps > max_dropout)[0]
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks + 1, [len(frames)]])
    raw = []
    for s_i, e_i in zip(starts, ends):
        run_f = frames[s_i:e_i]
        run_s = scores[s_i:e_i]
        raw.append({
            "start_frame": int(run_f[0]),
            "stop_frame": int(run_f[-1]),
            "n_hit": int(len(run_f)),
            "mean_score": float(np.mean(run_s)),
            "max_score": float(np.max(run_s)),
        })
    surv = [e for e in raw if (e["stop_frame"] - e["start_frame"] + 1) >= min_persist]
    if not surv:
        return []
    merged = [surv[0]]
    for e in surv[1:]:
        prev = merged[-1]
        if e["start_frame"] - prev["stop_frame"] <= merge_gap:
            tot = prev["n_hit"] + e["n_hit"]
            prev["mean_score"] = (prev["mean_score"] * prev["n_hit"] + e["mean_score"] * e["n_hit"]) / tot
            prev["max_score"] = max(prev["max_score"], e["max_score"])
            prev["n_hit"] = tot
            prev["stop_frame"] = e["stop_frame"]
        else:
            merged.append(e)
    return merged


def detect_circling_events(df_pairs, video_id=None, fps=FPS,
                           persistence_s=PERSISTENCE_S, dropout_s=DROPOUT_S, merge_s=MERGE_S,
                           behavior_label="circling"):
    """Extract circling event spans per pair episode."""
    if "rule_hit" not in df_pairs.columns:
        raise KeyError("df_pairs needs rule_hit; call add_rule_hit first.")
    min_persist = int(round(persistence_s * fps))
    max_dropout = int(round(dropout_s * fps))
    merge_gap = int(round(merge_s * fps))
    hits = df_pairs[df_pairs["rule_hit"]].sort_values(["pair_episode_id", "FrameNum"])
    rows = []
    for pid, grp in hits.groupby("pair_episode_id"):
        evs = _events_from_hits(grp["FrameNum"].to_numpy(), grp["rule_score"].to_numpy(),
                                min_persist, max_dropout, merge_gap)
        for ev in evs:
            rows.append({
                "pair_episode_id": int(pid),
                "TrackUID_1": int(grp["TrackUID_1"].iloc[0]),
                "TrackUID_2": int(grp["TrackUID_2"].iloc[0]),
                "sex_1": grp["sex_1"].iloc[0],
                "sex_2": grp["sex_2"].iloc[0],
                "start_frame": ev["start_frame"],
                "stop_frame": ev["stop_frame"],
                "n_hit": ev["n_hit"],
                "duration_s": round((ev["stop_frame"] - ev["start_frame"] + 1) / fps, 3),
                "mean_score": round(ev["mean_score"], 4),
                "max_score": round(ev["max_score"], 4),
            })
    analytical_cols = ["pair_episode_id", "TrackUID_1", "TrackUID_2", "sex_1", "sex_2",
                       "start_frame", "stop_frame", "n_hit", "duration_s", "max_score"]
    if not rows:
        return pd.DataFrame(columns=EVENT_COLS + analytical_cols)
    df = pd.DataFrame(rows).sort_values("start_frame").reset_index(drop=True)
    df = finalize_events_df(df, video_id=video_id, behavior_label=behavior_label, fps=fps,
                            fish_ids_cols=("TrackUID_1", "TrackUID_2"), confidence_col="mean_score")
    return df[EVENT_COLS + analytical_cols]


def plot_events_timeline(events_df, labels_df=None, fps=FPS, video_duration_s=None, save_path=None):
    """Timeline with detected events as blue bars and optional CSV label ticks."""
    import seaborn as sns
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(14, 3))
    for _, ev in events_df.iterrows():
        ax.plot([ev.start_frame / fps / 60, ev.stop_frame / fps / 60], [1, 1],
                color="steelblue", lw=20, solid_capstyle="butt")
    if labels_df is not None:
        for _, lab in labels_df.iterrows():
            mid_min = (lab.StartFrame + lab.StopFrame) / 2 / fps / 60
            ax.vlines(mid_min, -0.3, 0.3, color="crimson", lw=1.25, zorder=5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["labels (CSV)", "detected"])
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("time (min)")
    if video_duration_s:
        ax.set_xlim(0, video_duration_s / 60)
    title = f"detected={len(events_df)}"
    if labels_df is not None:
        title += f", labels={len(labels_df)}"
    ax.set_title(f"Circling events timeline ({title})")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


def plot_events_rich_timeline(events_df, labels_df=None, fps=FPS, video_duration_s=None, title=None, save_path=None):
    """Stem plot of event starts (height=duration, color=confidence) plus KDE density and optional CSV labels."""
    import seaborn as sns
    import matplotlib.ticker as mticker
    from matplotlib.colors import Normalize
    sns.set_style("whitegrid")
    starts_hr = events_df["start_frame"].to_numpy() / fps / 3600
    durs_s = events_df["duration_s"].to_numpy()
    confs = events_df["confidence"].to_numpy()
    xmax_hr = (video_duration_s or 10 * 3600) / 3600
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    norm = Normalize(vmin=max(float(confs.min()), 0.0), vmax=max(float(confs.max()), 1e-6))
    cmap = plt.get_cmap("plasma")
    for xi, hi, ci in zip(starts_hr, durs_s, confs):
        ax_top.vlines(xi, 0, hi, color=cmap(norm(ci)), lw=1.5, alpha=0.9)
    ax_top.scatter(starts_hr, durs_s, c=confs, cmap="plasma", norm=norm, s=40, edgecolor="black", linewidth=0.4, zorder=3)
    if labels_df is not None:
        lbl_starts_hr = labels_df["StartFrame"].to_numpy() / fps / 3600
        ax_top.scatter(lbl_starts_hr, [0] * len(lbl_starts_hr), marker="X", color="crimson", s=80, zorder=4, label="CSV label")
        ax_top.legend(loc="upper right")
    ax_top.set_ylabel("duration (s)")
    ax_top.set_title(title or f"Detected circling events (n={len(events_df)})")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=ax_top, pad=0.01, fraction=0.03, label="confidence")
    if len(starts_hr) > 1:
        sns.kdeplot(x=starts_hr, ax=ax_bot, fill=True, color="#9b1b67", alpha=0.6, bw_adjust=0.5, clip=(0, xmax_hr))
    ax_bot.set_ylabel("density")
    ax_bot.set_xlabel("time (hr)")
    ax_bot.set_xlim(0, xmax_hr)
    ax_bot.xaxis.set_major_locator(mticker.MultipleLocator(0.5))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()

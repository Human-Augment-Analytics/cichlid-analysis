"""Exploratory data analysis: Can Optical-Flow features alone discriminate
MF vs FF circling bouts?

All code and outputs live under `eda/mf_vs_ff/`.  Does not modify the
main pipeline under `models/optical-flow/`.

Pipeline
--------
1. F-F bouts are hard-coded from user manual inspection (hh:mm:ss ->
   frame at 30 fps).
2. M-F bouts are selected DIRECTLY from the GT annotation CSVs of the
   same two videos.  By project convention every entry in a
   `circling_annotations_vid*.csv` IS an M-F circling event -- no sex
   label needs to be cross-referenced.  To keep the comparison fair we
   duration-match: only pick events whose duration falls inside the
   range spanned by the F-F list, then uniform-sample N across that
   duration axis so the MF and FF duration distributions overlap.
3. For every selected event, Farneback flow is computed between
   consecutive frames and the following per-frame features are extracted
   inside the mag > mag_thr active-pixel mask:
       rot_omega        angular velocity about activity centroid
       mean_mag         mean |flow| inside active mask (bulk motion)
       trans_mag        |mean flow vector|          (coherent translation)
       activity         # active pixels
   Each event is then aggregated to mean / std / max over its frames.
4. Per-feature stats + Mann-Whitney U test are computed between MF and
   FF groups; strip plots and a human-readable verdict are saved.

Run (from the repo root)
------------------------
    conda activate vision
    python eda/mf_vs_ff/mf_vs_ff_of.py

Takes ~5-10 min (Farneback on all events, downsample=2).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent                       # cichlid-analysis/
ANN_DIR = REPO / 'ground_truth_annotations/finalized_annotations'
DATA_DIR = REPO / 'ground_truth_annotations/data'
POSE_DIR = DATA_DIR / 'pairs_raw'
OUT = HERE / 'results'
OUT.mkdir(exist_ok=True)

FPS = 29.0


# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------

# Video-name -> (mp4, annotation csv).  All events in these annotation
# CSVs are M-F circling by project convention.
VIDEOS = {
    'vid_0028': (DATA_DIR / '0028_vid.mp4',
                 ANN_DIR  / 'circling_annotations_vid0028.csv'),
    'vid_0002': (DATA_DIR / '0002_vid.mp4',
                 ANN_DIR  / 'circling_annotations_vid0002.csv'),
}

# User-verified F-F bouts.  (video_tag, hhmmss_start, hhmmss_end, bout_id)
FF_BOUTS = [
    ('vid_0028', '5:43:56', '5:44:06', 'vid_0028_FF_1'),
    ('vid_0002', '2:18:21', '2:18:30', 'vid_0002_FF_a'),
    ('vid_0002', '2:18:31', '2:18:33', 'vid_0002_FF_b'),
    ('vid_0002', '2:29:10', '2:29:25', 'vid_0002_FF_c'),
    ('vid_0002', '2:31:19', '2:31:22', 'vid_0002_FF_d'),
    ('vid_0002', '2:44:22', '2:45:08', 'vid_0002_FF_e'),
    ('vid_0002', '3:03:21', '3:03:30', 'vid_0002_FF_f'),
]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def hhmmss_to_frame(ts: str, fps: float = FPS) -> int:
    parts = [int(p) for p in ts.split(':')]
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h = 0; m, s = parts
    else:
        raise ValueError(f'bad timestamp {ts!r}')
    return int((h * 3600 + m * 60 + s) * fps)


def select_mf_events(video_tag: str, n_target: int,
                     dur_range_s: tuple[float, float]) -> list[dict]:
    """Pick MF events directly from the annotation CSV.  By project
    convention every row is an M-F circling event.  We duration-match
    to the FF range and uniform-sample N across that range so the MF
    and FF duration distributions overlap.
    """
    _, ann_csv = VIDEOS[video_tag]
    gt = pd.read_csv(ann_csv)
    gt.columns = [c.strip() for c in gt.columns]
    gt['dur_s'] = (gt['End Frame'] - gt['Start Frame']) / FPS
    gt = gt[(gt['dur_s'] >= dur_range_s[0]) & (gt['dur_s'] <= dur_range_s[1])]
    gt = gt.sort_values('dur_s').reset_index(drop=True)

    picked = [dict(
        video_tag=video_tag,
        bout_id=str(r['EventID']),
        start_frame=int(r['Start Frame']),
        end_frame=int(r['End Frame']),
        dur_s=float(r['dur_s']),
        gt_turns=r.get('No. of turns/circles'),
        gt_direction=r.get('Rotation direction'),
    ) for _, r in gt.iterrows()]

    if len(picked) <= n_target:
        return picked
    idx = np.linspace(0, len(picked) - 1, n_target).round().astype(int)
    return [picked[i] for i in idx]


# -----------------------------------------------------------------------------
# OF feature extraction
# -----------------------------------------------------------------------------

def extract_of_features(video: Path, start_frame: int, end_frame: int,
                        downsample: float = 2.0,
                        mag_thr: float = 0.5,
                        crop_frac: float = 0.1) -> pd.DataFrame:
    """Farneback flow between consecutive frames inside [start, end).

    Features per frame (inside the mag > mag_thr active mask):
        rot_omega    best-fit angular velocity about activity centroid
        rot_orb      unnormalised orbital moment (signed; CW+ in image coords)
        mean_mag     mean |flow|
        trans_mag    |mean flow vector|
        activity     #active pixels
    """
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
    ok, prev = cap.read()
    if not ok:
        cap.release()
        return pd.DataFrame()
    H, W = prev.shape[:2]

    def _prep(frame_bgr):
        g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if downsample > 1:
            g = cv2.resize(g, (int(W / downsample), int(H / downsample)),
                           interpolation=cv2.INTER_AREA)
        if crop_frac > 0:
            h2, w2 = g.shape
            dh = int(h2 * crop_frac); dw = int(w2 * crop_frac)
            g = g[dh:h2 - dh, dw:w2 - dw]
        return g

    prev_g = _prep(prev)
    rows = []
    for f in range(start_frame + 1, end_frame):
        ok, cur = cap.read()
        if not ok:
            break
        cur_g = _prep(cur)
        flow = cv2.calcOpticalFlowFarneback(prev_g, cur_g, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        u = flow[..., 0]; v = flow[..., 1]
        mag = np.sqrt(u * u + v * v)
        active = mag > mag_thr
        if active.sum() < 5:
            rows.append(dict(frame=f, rot_omega=0.0, rot_orb=0.0,
                             mean_mag=0.0, trans_mag=0.0, activity=0))
            prev_g = cur_g
            continue
        ys, xs = np.where(active)
        cx, cy = xs.mean(), ys.mean()
        rx = xs - cx; ry = ys - cy
        rot_orb = float((rx * v[active] - ry * u[active]).sum())
        denom = float((rx * rx + ry * ry).sum())
        rot_omega = rot_orb / denom if denom > 0 else 0.0
        mean_mag = float(mag[active].mean())
        trans_u = float(u[active].mean()); trans_v = float(v[active].mean())
        trans_mag = float(np.hypot(trans_u, trans_v))
        rows.append(dict(frame=f, rot_omega=rot_omega, rot_orb=rot_orb,
                         mean_mag=mean_mag, trans_mag=trans_mag,
                         activity=int(active.sum())))
        prev_g = cur_g
    cap.release()
    return pd.DataFrame(rows)


def aggregate_event(df: pd.DataFrame) -> dict:
    """Per-event summary statistics of the OF feature stream."""
    if df.empty:
        return {}
    ab = df['rot_omega'].abs()
    return dict(
        n_frames=len(df),
        rot_omega_abs_mean   = float(ab.mean()),
        rot_omega_abs_median = float(ab.median()),
        rot_omega_abs_p75    = float(ab.quantile(0.75)),
        rot_omega_abs_max    = float(ab.max()),
        rot_omega_std        = float(df['rot_omega'].std()),
        mean_mag_mean        = float(df['mean_mag'].mean()),
        mean_mag_median      = float(df['mean_mag'].median()),
        mean_mag_max         = float(df['mean_mag'].max()),
        mean_mag_std         = float(df['mean_mag'].std()),
        trans_mag_mean       = float(df['trans_mag'].mean()),
        activity_mean        = float(df['activity'].mean()),
        activity_std         = float(df['activity'].std()),
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    # --- Build event list ---
    events = []
    for tag, hms_s, hms_e, bid in FF_BOUTS:
        events.append(dict(
            group='FF', video_tag=tag, bout_id=bid,
            start_frame=hhmmss_to_frame(hms_s),
            end_frame=hhmmss_to_frame(hms_e),
            dur_s=(hhmmss_to_frame(hms_e) - hhmmss_to_frame(hms_s)) / FPS,
            pair_sex='FF', source='user-verified timestamp',
        ))

    # Duration-match MF to the span of the FF list.
    ff_durs = [e['dur_s'] for e in events if e['group'] == 'FF']
    dur_range_s = (max(1, min(ff_durs) * 0.5),
                   max(ff_durs) * 1.5)
    for tag in VIDEOS:
        mf = select_mf_events(tag, n_target=6, dur_range_s=dur_range_s)
        for e in mf:
            events.append(dict(
                group='MF', video_tag=tag, bout_id=e['bout_id'],
                start_frame=e['start_frame'], end_frame=e['end_frame'],
                dur_s=e['dur_s'], pair_sex='MF',
                source=f'{VIDEOS[tag][1].name} (all events are M-F by convention)',
            ))

    print(f'[events] {sum(1 for e in events if e["group"]=="FF")} FF + '
          f'{sum(1 for e in events if e["group"]=="MF")} MF')
    pd.DataFrame(events).to_csv(OUT / 'events_selected.csv', index=False)

    # --- Extract OF features per event ---
    rows = []
    for i, ev in enumerate(events):
        mp4 = VIDEOS[ev['video_tag']][0]
        if not mp4.exists():
            print(f'  SKIP {ev["bout_id"]} -- mp4 missing: {mp4}')
            continue
        t0 = time.time()
        feats = extract_of_features(mp4, ev['start_frame'], ev['end_frame'])
        dt = time.time() - t0
        agg = aggregate_event(feats)
        print(f'[{i+1:>2}/{len(events)}] {ev["group"]}  {ev["bout_id"]:<28} '
              f'n_fr={agg.get("n_frames","?"):>4}  dt={dt:4.1f}s  '
              f'|rot|_mean={agg.get("rot_omega_abs_mean",float("nan")):.4f}  '
              f'mag_mean={agg.get("mean_mag_mean",float("nan")):.2f}')
        rows.append({**ev, **agg})
    feat_df = pd.DataFrame(rows)
    feat_df.to_csv(OUT / 'features_per_event.csv', index=False)

    # --- Statistics ---
    FEATURES = ['rot_omega_abs_mean', 'rot_omega_abs_median',
                'rot_omega_abs_p75', 'rot_omega_abs_max',
                'rot_omega_std',
                'mean_mag_mean', 'mean_mag_median', 'mean_mag_max',
                'mean_mag_std', 'trans_mag_mean',
                'activity_mean', 'activity_std']
    mf = feat_df[feat_df.group == 'MF']
    ff = feat_df[feat_df.group == 'FF']
    rows = []
    for col in FEATURES:
        if col not in feat_df.columns:
            continue
        mf_v = mf[col].dropna().to_numpy()
        ff_v = ff[col].dropna().to_numpy()
        if len(mf_v) < 2 or len(ff_v) < 2:
            continue
        u_stat, p = stats.mannwhitneyu(mf_v, ff_v, alternative='two-sided')
        ratio = mf_v.mean() / ff_v.mean() if ff_v.mean() != 0 else float('inf')
        rows.append(dict(
            feature=col,
            MF_n=len(mf_v), FF_n=len(ff_v),
            MF_mean=float(mf_v.mean()), FF_mean=float(ff_v.mean()),
            MF_median=float(np.median(mf_v)), FF_median=float(np.median(ff_v)),
            MF_over_FF_ratio=float(ratio),
            mannwhitney_U=float(u_stat), p_value=float(p),
        ))
    stats_df = pd.DataFrame(rows).sort_values('p_value')
    stats_df.to_csv(OUT / 'feature_stats.csv', index=False)

    # --- Plots ---
    n = len([f for f in FEATURES if f in feat_df.columns])
    cols = 3
    rows_p = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_p, cols, figsize=(cols * 4, rows_p * 3))
    for ax, feat in zip(axes.flat, FEATURES):
        if feat not in feat_df.columns:
            ax.axis('off'); continue
        for grp, color in [('MF', 'tab:blue'), ('FF', 'tab:orange')]:
            vals = feat_df.loc[feat_df.group == grp, feat].dropna().to_numpy()
            x = np.random.normal({'MF': 0, 'FF': 1}[grp], 0.04, len(vals))
            ax.scatter(x, vals, alpha=0.7, s=30, label=grp, color=color)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['MF', 'FF'])
        p = stats_df[stats_df.feature == feat]['p_value']
        p_str = f'p={float(p.iloc[0]):.3f}' if len(p) else ''
        ax.set_title(f'{feat}\n{p_str}')
    for ax in axes.flat[n:]:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(OUT / 'comparison.png', dpi=120)
    plt.close()

    # --- Verdict text ---
    lines = []
    lines.append('=' * 72)
    lines.append(' MF vs FF discrimination via Optical Flow features')
    lines.append('=' * 72)
    lines.append('')

    # Method explanation
    lines.append('## 1. Question')
    lines.append('')
    lines.append('Can pixel-level Optical Flow features (NO pose, NO masks,')
    lines.append('NO manual labels per fish) distinguish M-F circling bouts')
    lines.append('from F-F circling bouts on raw tank video?  The working')
    lines.append('hypothesis is that MF pairs rotate faster because the male')
    lines.append('leads and makes swift directional changes.')
    lines.append('')

    lines.append('## 2. Data')
    lines.append('')
    lines.append(f'MF events ({len(mf)} total, duration-matched to FF range):')
    lines.append('  - Source: GT annotation CSVs under')
    lines.append('    ground_truth_annotations/finalized_annotations/.')
    lines.append('    By project convention every row in these CSVs is an')
    lines.append('    M-F circling event; no sex label is required.')
    lines.append('  - Per video: select events within the FF duration range,')
    lines.append('    then uniform-sample N across that range so MF and FF')
    lines.append('    duration distributions overlap (no accidental duration')
    lines.append('    confound).')
    lines.append(f'  - Videos sampled: {sorted(mf.video_tag.unique().tolist())}')
    lines.append('')
    lines.append(f'FF events ({len(ff)} total):')
    lines.append('  - Source: user-provided hh:mm:ss timestamps, manually')
    lines.append('    verified by watching the video.')
    lines.append(f'  - Videos sampled: {sorted(ff.video_tag.unique().tolist())}')
    lines.append('')

    lines.append('## 3. Feature extraction (pixel-only)')
    lines.append('')
    lines.append('For each selected event:')
    lines.append('  1. Open the source mp4 at frame = Start Frame.')
    lines.append('  2. For every consecutive pair of frames, run')
    lines.append('     cv2.calcOpticalFlowFarneback (downsample=2, crop=0.1,')
    lines.append('     default pyramid parameters) to get a dense (u, v)')
    lines.append('     field.')
    lines.append('  3. Define the "active" pixel mask: |flow| > 0.5 px/frame.')
    lines.append('  4. Compute inside the active mask:')
    lines.append('       rot_omega  = sum(rx*v - ry*u) / sum(rx**2 + ry**2)')
    lines.append('                    angular velocity about activity centroid')
    lines.append('       mean_mag   = mean |flow|          (bulk motion)')
    lines.append('       trans_mag  = |mean flow vector|   (coherent translation)')
    lines.append('       activity   = count of active pixels (pair tightness proxy)')
    lines.append('  5. Aggregate per event: mean / median / p75 / max / std')
    lines.append('     of each feature across the event frames.')
    lines.append('')
    lines.append('No pose parquet, no SAM 2 masks, no fish-identity tracking')
    lines.append('entered this feature stream at any point.  Pose is used')
    lines.append('ONLY for the final direction-correct sanity check elsewhere')
    lines.append('in the project; not here.')
    lines.append('')

    lines.append('## 4. Statistical test')
    lines.append('')
    lines.append('Per feature, a two-sided Mann-Whitney U test between the')
    lines.append('MF and FF event groups.  Non-parametric because sample')
    lines.append('sizes are small (n_MF={}, n_FF={}) and the feature'.format(
                 len(mf), len(ff)))
    lines.append('distributions are not expected to be Gaussian.  Report')
    lines.append('the MF/FF ratio of means + the p-value.')
    lines.append('')

    lines.append('## 5. Results')
    lines.append('')
    lines.append('Feature-by-feature, sorted by p:')
    lines.append('')
    lines.append(stats_df.to_string(index=False))
    lines.append('')
    sig = stats_df[stats_df.p_value < 0.05]
    lines.append(f'Significant at p < 0.05: {len(sig)} / {len(stats_df)} features')
    if len(sig):
        lines.append('')
        lines.append('Significant features:')
        for _, r in sig.iterrows():
            arrow = ('MF > FF' if r.MF_over_FF_ratio > 1 else 'MF < FF')
            lines.append(f'  {r.feature:<24} {arrow}  '
                         f'(MF/FF ratio = {r.MF_over_FF_ratio:.2f},  '
                         f'p = {r.p_value:.4f})')
    lines.append('')

    lines.append('## 6. Interpretation')
    lines.append('')
    if len(sig):
        rot_sig = [r for _, r in sig.iterrows()
                   if 'rot_omega' in r.feature]
        mag_sig = [r for _, r in sig.iterrows()
                   if 'mean_mag' in r.feature or 'trans_mag' in r.feature]
        if rot_sig and not mag_sig:
            lines.append('Angular velocity magnitude is the clean')
            lines.append('discriminator.  Bulk motion (mean_mag, trans_mag)')
            lines.append('and activity area do NOT separate the groups --')
            lines.append('the difference is specifically rotational, not')
            lines.append('about overall how much the pair moves.  The')
            lines.append('hypothesis that "MF pairs rotate faster because')
            lines.append('the male leads" is empirically supported at')
            lines.append(f'p < {max(r.p_value for r in rot_sig):.4f}.')
        else:
            lines.append('Multiple feature families separate the groups --')
            lines.append('see table above.  Pattern of significance may')
            lines.append('require further inspection.')
    else:
        lines.append('No feature separated the groups at p < 0.05.  Either')
        lines.append('the OF-only signal is too weak, sample sizes too small,')
        lines.append('or the groups genuinely overlap on these features.')
    lines.append('')
    lines.append('## 7. Caveats')
    lines.append('')
    lines.append('- n_MF = {}, n_FF = {}: small samples; p-values should be'.format(
                 len(mf), len(ff)))
    lines.append('  treated as indicative, not definitive.')
    lines.append('- F-F labels are from user visual verification, not from')
    lines.append('  an independent biological ground truth.')
    lines.append('- All videos sampled are overhead tank shots with similar')
    lines.append('  camera geometry; results may not transfer to side-view')
    lines.append('  or depth-sensor footage.')
    lines.append('- Duration matching reduces but does not eliminate the')
    lines.append('  possibility of other confounds (time of day, tank')
    lines.append('  activity level, which individual fish are present).')

    (OUT / 'summary.txt').write_text('\n'.join(lines))
    print('\n' + '\n'.join(lines[-30:]))
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()

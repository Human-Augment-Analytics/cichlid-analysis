"""Aggregate per-video + overall MAE/RMSE/direction stats for pose num_turns outputs."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        '/mnt/c/Users/thaku/OneDrive/Desktop/Rupak/OMSCS/CS8903-Biomedical AI/Cichlid project/HAAG repo/outputs/optical-flow/pose')
    vids = ['vid0002', 'vid0007-day5', 'vid0007-day6', 'vid013',
            'vid0018', 'vid0028', 'vid0031', 'vid0037']
    summary_lines = []
    def _emit(line=''):
        print(line)
        summary_lines.append(line)

    rows_per_video = []
    all_rows = []

    header = f'{"video":<16}{"n":>5}{"dir%":>7}{"exact%":>8}{"<=0.5%":>8}{"<=1%":>7}{"<=2%":>7}{"MAE_q":>8}{"RMSE_q":>8}{"MAE_raw":>9}{"RMSE_raw":>10}'
    _emit(header)
    _emit('-' * len(header))

    for v in vids:
        p = root / f'{v}_events.csv'
        if not p.exists():
            _emit(f'{v} MISSING')
            continue
        df = pd.read_csv(p)
        df['video'] = v
        scored = df.dropna(subset=['pred_turns_quantized'])
        if scored.empty:
            continue
        gt = scored.gt_turns
        q = scored.pred_turns_quantized.abs()
        raw = scored.pred_turns_raw.abs()
        dir_pct   = 100 * (scored.pred_direction == scored.gt_direction).mean()
        exact_pct = 100 * (q == gt).mean()
        w05_pct   = 100 * (np.abs(q - gt) <= 0.5).mean()
        w1_pct    = 100 * (np.abs(q - gt) <= 1).mean()
        w2_pct    = 100 * (np.abs(q - gt) <= 2).mean()
        mae_q     = np.mean(np.abs(q - gt))
        rmse_q    = np.sqrt(np.mean((q - gt) ** 2))
        mae_raw   = np.mean(np.abs(raw - gt))
        rmse_raw  = np.sqrt(np.mean((raw - gt) ** 2))
        rows_per_video.append(dict(
            video=v, n=len(scored),
            dir_pct=dir_pct, exact_pct=exact_pct,
            w05_pct=w05_pct, w1_pct=w1_pct, w2_pct=w2_pct,
            mae_q=mae_q, rmse_q=rmse_q,
            mae_raw=mae_raw, rmse_raw=rmse_raw,
        ))
        all_rows.append(scored)
        _emit(f'{v:<16}{len(scored):>5}'
              f'{dir_pct:>6.1f}%{exact_pct:>7.1f}%{w05_pct:>7.1f}%'
              f'{w1_pct:>6.1f}%{w2_pct:>6.1f}%'
              f'{mae_q:>8.2f}{rmse_q:>8.2f}{mae_raw:>9.2f}{rmse_raw:>10.2f}')

    _emit('\n=== OVERALL (all 8 videos combined) ===')
    ad = pd.concat(all_rows)
    gt = ad.gt_turns
    q = ad.pred_turns_quantized.abs()
    raw = ad.pred_turns_raw.abs()
    _emit(f'events scored:  {len(ad)}')
    _emit(f'direction:      {(ad.pred_direction == ad.gt_direction).sum()}/{len(ad)} = '
          f'{100*(ad.pred_direction == ad.gt_direction).mean():.1f}%')
    _emit(f'exact match:    {100 * (q == gt).mean():.1f}%')
    _emit(f'within 0.5:     {100 * (np.abs(q - gt) <= 0.5).mean():.1f}%')
    _emit(f'within 1:       {100 * (np.abs(q - gt) <= 1).mean():.1f}%')
    _emit(f'within 2:       {100 * (np.abs(q - gt) <= 2).mean():.1f}%')
    _emit(f'MAE  quantized: {np.mean(np.abs(q - gt)):.3f} turns')
    _emit(f'RMSE quantized: {np.sqrt(np.mean((q - gt) ** 2)):.3f} turns')
    _emit(f'MAE  raw:       {np.mean(np.abs(raw - gt)):.3f} turns')
    _emit(f'RMSE raw:       {np.sqrt(np.mean((raw - gt) ** 2)):.3f} turns')

    err = q - gt
    _emit(f'\nquantized error: mean={err.mean():+.3f}  median={err.median():+.2f}  '
          f'min={err.min():+.1f}  max={err.max():+.1f}')
    _emit(f'\nerror distribution:')
    for v in sorted(err.round(1).unique()):
        c = int((err.round(1) == v).sum())
        p = 100 * c / len(err)
        if c > 0:
            _emit(f'  err={v:+5.1f}: {c:>4d} ({p:5.1f}%)')

    # ---- Dump to outputs ----
    # per-video summary CSV
    if rows_per_video:
        pv = pd.DataFrame(rows_per_video)
        pv.to_csv(root / 'summary_per_video.csv', index=False)
        _emit(f'\nwrote {root / "summary_per_video.csv"}')
    # concatenated events CSV (every event from every video, with video id)
    if all_rows:
        ad = pd.concat(all_rows, ignore_index=True)
        ad.to_csv(root / 'all_events_combined.csv', index=False)
        _emit(f'wrote {root / "all_events_combined.csv"}')
    # human-readable summary txt
    (root / 'summary.txt').write_text('\n'.join(summary_lines))
    _emit(f'wrote {root / "summary.txt"}')


if __name__ == '__main__':
    main()

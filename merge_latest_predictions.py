"""Merge finalized ground truth with committed model predictions for fresh review.

Existing manually verified merged_spans CSVs are never overwritten. Ground-truth
whole-second times are mapped to inclusive frame endpoints at nominal 30 FPS,
matching the source annotation export's start_frame/stop_frame convention.
The timestamps are quantized; exact original subsecond boundaries are unavailable.
"""
import argparse
import csv
import io
import json
import subprocess
from decimal import Decimal
from pathlib import Path

from merge_intervals import ROOT, Interval, merge_intervals

MODELS = {'pose': 'outputs/dlc2action_test/predicted_segments.csv',
          'videomae': 'outputs/videomae_mstcn/predicted_segments.csv'}


def time_frame(value, fps):
    h, m, s = value.strip().split(':')
    return int((Decimal(h)*3600 + Decimal(m)*60 + Decimal(s))*Decimal(str(fps)))


def merge_latest(commit, models, output_dir, fps=30):
    sha = subprocess.check_output(['git', 'rev-parse', '--verify', f'{commit}^{{commit}}'], cwd=ROOT, text=True).strip()
    output_dir = Path(output_dir)
    if output_dir.resolve() == (ROOT/'outputs'/'merged_spans').resolve():
        raise ValueError('Choose a separate review directory to preserve manually verified spans')
    intervals, provenance, counts = [], [], {}
    for path in sorted((ROOT/'ground-truth-annotations'/'finalized-annotations').glob('*.csv')):
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        counts[path.name] = len(rows)
        for index, row in enumerate(rows, 2):
            if row['Behavior'].strip().lower() != 'circling':
                raise ValueError(f'Unexpected behavior at {path}:{index}')
            # VideoID is authoritative: some 0031 EventIDs incorrectly start with 0028.
            source_id = f"ground-truth:{path.name}:{row['EventID']}"
            start, end = time_frame(row['StartTime'], fps), time_frame(row['EndTime'], fps)
            intervals.append(Interval(row['VideoID'], start, end, source_id, 'rule-based'))
            provenance.append(dict(source_id=source_id, source='ground-truth', video=row['VideoID'],
                start_frame=start, end_frame=end, file=str(path.relative_to(ROOT)), row=index,
                original_event_id=row['EventID'], fish_ids=row['FishIDs'], notes=row['Confidence/Notes']))
    if not intervals:
        raise ValueError('No finalized annotations found')
    for model in models:
        path = MODELS[model]
        content = subprocess.check_output(['git', 'show', f'{sha}:{path}'], cwd=ROOT, text=True)
        rows = list(csv.DictReader(io.StringIO(content)))
        counts[model] = len(rows)
        for index, row in enumerate(rows, 2):
            start, end = int(row['start_frame']), int(row['end_frame'])
            if int(row['frames']) != end-start+1:
                raise ValueError(f'Invalid inclusive duration: {model} row {index}')
            source_id = f"{model}:{row['video']}:{start}-{end}"
            intervals.append(Interval(row['video'], start, end, source_id, 'predicted'))
            provenance.append(dict(source_id=source_id, source=model, video=row['video'],
                start_frame=start, end_frame=end, file=path, commit=sha, row=index,
                sequence=row['sequence'], mean_score=float(row['mean_score'])))
    merged = merge_intervals(intervals, fps)
    output_dir.mkdir(parents=True, exist_ok=True)
    for video, rows in merged.items():
        for row in rows:
            row['Source'] = {'rule-based': 'ground-truth', 'both': 'both', 'predicted': 'predicted'}[row['Source']]
            row['Ground Truth IDs'] = row.pop('Rule IDs')
            # Fresh candidates must be reviewed, even if part overlaps accepted ground truth.
            row['Verified'] = ''
        with (output_dir/f'{video}_merged_spans.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    info = {'prediction_commit': sha, 'models': models, 'input_counts': counts,
            'merged_counts': {v: len(rows) for v, rows in merged.items()}, 'fps': fps,
            'merge_policy': 'Union overlapping or adjacent inclusive intervals, separately per VideoID.',
            'verification_policy': 'Fresh review rows have blank Verified; original manually verified CSVs are unchanged.',
            'time_policy': 'Ground-truth whole-second timestamps multiplied by nominal FPS. Both endpoints inclusive, '
                           'following source start_frame/stop_frame export. Original subsecond precision is unavailable.',
            'prediction_scope': 'Only videos present in committed predictions; absence is not a negative label.',
            'inputs': provenance}
    (output_dir/'provenance.json').write_text(json.dumps(info, indent=2)+'\n')
    (output_dir/'README.md').write_text(
        '# Ground truth and latest model predictions\n\n'
        f'Prediction commit: `{sha}`. Models: {", ".join(models)}.\n\n'
        + '\n'.join(f'- {v}: {len(rows)} merged review spans.' for v, rows in merged.items())
        + '\n\n' + info['merge_policy'] + '\n\n' + info['time_policy'] + '\n\n'
        + info['verification_policy'] + '\n\n'
        + 'Ground Truth IDs and Predicted Event IDs retain all contributors. `provenance.json` '
          'records original rows, model scores, source files and commit. VideoID overrides mismatched '
          'prefixes in ground-truth EventIDs. No duration or confidence filters are added.\n')
    print(json.dumps({k:v for k,v in info.items() if k!='inputs'},indent=2))
    return merged


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--models', nargs='+', choices=list(MODELS), required=True)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs'/'merged_spans'/'latest_model_review')
    args = parser.parse_args()
    merge_latest(args.commit, args.models, args.output_dir)

"""Merge rule-based and predicted circling spans by video.

Frame ranges are zero-based and inclusive. Overlapping or adjacent ranges form
one merged span, and the output records every contributing source ID.
"""

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_FIELDS = (
    'Video ID', 'Merge ID', 'Start Frame', 'End Frame', 'Verified', 'Start Time', 'End Time',
    'Source', 'Rule IDs', 'Predicted Event IDs',
)


@dataclass(frozen=True)
class Interval:
    video_id: str
    start: int
    end: int
    source_id: str
    source: str

    def __post_init__(self):
        if self.start < 0 or self.end < self.start:
            raise ValueError(f'Invalid interval: {self}')


def read_rule_intervals(path):
    with Path(path).open(newline='') as stream:
        for line, row in enumerate(csv.reader(stream, delimiter='\t'), 1):
            if not row:
                continue
            if len(row) != 3 or not re.fullmatch(r'\d{4}-\d+', row[0]):
                raise ValueError(f'Invalid rule-based row {line}: {row}')
            rule_id, start, end = row
            yield Interval(f'{rule_id[:4]}_vid', int(start), int(end), rule_id, 'rule-based')


def read_predicted_intervals(paths):
    for path in paths:
        with Path(path).open(newline='') as stream:
            for row in csv.DictReader(stream):
                yield Interval(
                    row['Video ID'], int(row['Start Frame']), int(row['End Frame']),
                    row['Event ID'], 'predicted',
                )


def frame_time(frame, fps):
    milliseconds = round(frame * 1000 / fps)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, remainder = divmod(remainder, 1_000)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}.{remainder:03d}'


def merge_intervals(intervals, fps=30.):
    """Return sorted, disjoint rows grouped by video, keeping source IDs."""
    if fps <= 0:
        raise ValueError('fps must be positive')
    grouped = defaultdict(list)
    for interval in intervals:
        grouped[interval.video_id].append(interval)

    result = {}
    for video_id, video_intervals in sorted(grouped.items()):
        merged = []
        for interval in sorted(video_intervals, key=lambda item: (item.start, item.end)):
            if merged and interval.start <= merged[-1]['end'] + 1:
                group = merged[-1]
                group['end'] = max(group['end'], interval.end)
            else:
                group = {'start': interval.start, 'end': interval.end,
                         'rule_ids': [], 'predicted_ids': []}
                merged.append(group)
            ids = group['rule_ids' if interval.source == 'rule-based' else 'predicted_ids']
            if interval.source_id not in ids:
                ids.append(interval.source_id)

        rows = []
        for number, group in enumerate(merged, 1):
            start, end = group['start'], group['end']
            rule_ids, predicted_ids = group['rule_ids'], group['predicted_ids']
            source = 'both' if rule_ids and predicted_ids else (
                'rule-based' if rule_ids else 'predicted')
            rows.append({
                'Video ID': video_id,
                'Merge ID': f'{video_id}-{number:04d}',
                'Start Frame': start,
                'End Frame': end,
                'Verified': '',
                'Start Time': frame_time(start, fps),
                'End Time': frame_time(end + 1, fps),
                'Source': source,
                'Rule IDs': ', '.join(rule_ids),
                'Predicted Event IDs': ', '.join(predicted_ids),
            })
        result[video_id] = rows
    return result


def write_merged_spans(rule_path, predicted_paths, output_dir, fps=30.):
    intervals = [*read_rule_intervals(rule_path), *read_predicted_intervals(predicted_paths)]
    if not intervals:
        raise ValueError('No input spans found')
    merged = merge_intervals(intervals, fps=fps)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for video_id, rows in merged.items():
        destination = output_dir / f'{video_id}_merged_spans.csv'
        verified_by_range = {}
        if destination.exists():
            with destination.open(newline='') as stream:
                for previous in csv.DictReader(stream):
                    if 'Verified' in previous:
                        key = (int(previous['Start Frame']), int(previous['End Frame']))
                        verified_by_range[key] = previous['Verified']
        for row in rows:
            row['Verified'] = verified_by_range.get(
                (row['Start Frame'], row['End Frame']), '')
        with destination.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        written.append(destination)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rules', type=Path, default=ROOT / 'rule-based.tsv')
    parser.add_argument('--predicted', nargs='+', type=Path,
                        help='Predicted span CSV files (default: both 0.50 files)')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs' / 'merged_spans')
    parser.add_argument('--fps', type=float, default=30.)
    args = parser.parse_args()
    predicted = args.predicted or sorted(
        (ROOT / 'outputs' / 'predicted_spans').glob('*_vid_spans_threshold_0.50.csv'))
    if not predicted:
        parser.error('No predicted span CSV files found')
    for path in write_merged_spans(args.rules, predicted, args.output_dir, args.fps):
        with path.open(newline='') as stream:
            count = sum(1 for _ in csv.DictReader(stream))
        print(f'{path}: {count} spans')


if __name__ == '__main__':
    main()

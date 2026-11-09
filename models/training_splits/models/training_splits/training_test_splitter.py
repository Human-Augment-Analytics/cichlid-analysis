import polars as pl
from pathlib import Path
import argparse


def split_video_data(df: pl.DataFrame, train_ranges: list[tuple[float, float]], test_ranges: list[tuple[float, float]],
                     gap_seconds: float) -> pl.DataFrame:
    """
    Split video data into training and test sets based on time windows, incorporating a gap at boundaries.

    Args:
        df: Polars DataFrame containing 'Time_s' and 'Circling'.
        train_ranges: List of (start, end) tuples for training in seconds.
        test_ranges: List of (start, end) tuples for testing in seconds.
        gap_seconds: Duration of the gap in seconds.

    Returns:
        DataFrame with an added 'set' column ('train', 'test', or null/buffer).
    """
    # Initialize mask
    train_mask = pl.lit(False)
    test_mask = pl.lit(False)

    # For each boundary, we need to apply the gap.
    # The simplest approach is to check if a range boundary is 'adjacent' to a boundary of the other set.
    # Given the train/test ranges, we can iterate through them and apply the gap to the boundaries.

    # 1. Apply gap to train ranges
    # If a train range ends at T, and there's a test range starting at T, we subtract gap_seconds from T.
    # This is a bit complex for a general case.
    # For now, let's implement the specific case requested:
    # Train: 0-6h (0-21600s), 8-10h (28800-36000s)
    # Test: 6-8h (21600-28800s)
    # Gap: 5 min (300s)

    # Train 1: [0, 21600] -> [0, 21600 - 300] = [0, 21300]
    # Test 1: [21600, 28800] -> [21600 + 300, 28800 - 300] = [21900, 28500]
    # Train 2: [28800, 36000] -> [28800 + 300, 36000] = [29100, 36000]

    # Let's apply a general logic:
    # For every pair of (train_range, test_range), if they are adjacent, apply gap.
    # Since we have the ranges, we can pre-process them.

    adjusted_train_ranges = []
    for start, end in train_ranges:
        # Check against test ranges
        for t_start, t_end in test_ranges:
            if abs(end - t_start) < 1:  # Adjacent at end
                end -= gap_seconds
            if abs(start - t_end) < 1:  # Adjacent at start
                start += gap_seconds
        adjusted_train_ranges.append((start, end))

    adjusted_test_ranges = []
    for start, end in test_ranges:
        # Check against train ranges
        for tr_start, tr_end in train_ranges:
            if abs(end - tr_start) < 1:  # Adjacent at end
                end -= gap_seconds
            if abs(start - tr_end) < 1:  # Adjacent at start
                start += gap_seconds
        adjusted_test_ranges.append((start, end))

    # Now create masks
    for start, end in adjusted_train_ranges:
        train_mask = train_mask | (pl.col("Time_s").is_between(start, end))

    for start, end in adjusted_test_ranges:
        test_mask = test_mask | (pl.col("Time_s").is_between(start, end))

    return df.with_columns(
        pl.when(train_mask).then(pl.lit("train"))
        .when(test_mask).then(pl.lit("test"))
        .otherwise(pl.lit(None))
        .alias("set")
    )


def main():
    parser = argparse.ArgumentParser(description="Split video parquet data into train/test sets.")
    parser.add_argument("input", help="Path to input parquet file")
    parser.add_argument("--output-dir", required=True, help="Directory to save splits")
    parser.add_argument("--gap", type=float, default=300, help="Gap in seconds at boundaries")

    args = parser.parse_args()

    df = pl.read_parquet(args.input)

    # Define ranges (hardcoded for now as per requirement)
    train_ranges = [(0, 21600), (28800, 36000)]  # 0-6h, 8-10h in seconds
    test_ranges = [(21600, 28800)]  # 6-8h in seconds

    split_df = split_video_data(df, train_ranges, test_ranges, args.gap)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Filter and save
    train_df = split_df.filter(pl.col("set") == "train")
    test_df = split_df.filter(pl.col("set") == "test")

    train_df.write_parquet(output_path / f"{Path(args.input).stem}_train.parquet")
    test_df.write_parquet(output_path / f"{Path(args.input).stem}_test.parquet")
    print(f"Saved splits to {output_path}")


if __name__ == "__main__":
    main()
## Example: python .\training_test_splitter.py "../../ground_truth_annotations/finalized_annotations/0028_vid_labeled.parquet" --output-dir "models/training_splits"
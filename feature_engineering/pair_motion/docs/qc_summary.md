# Data quality check

Checks run before building pair-level distance, orientation, and speed features.

| Video | Track rows | Pair rows | Track IDs | Pair episodes | Median episode |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0028_vid | 1,372,748 | 636,505 | 13,427 | 13,601 | 20 frames |
| 0031_vid | 1,320,026 | 553,836 | 11,001 | 9,830 | 22 frames |

## Results

- All 1,190,341 pair rows found both fish tracks.
- Nose and tail coordinates are complete; `qc_n_missing_kp` is zero in both videos.
- Pair ordering is valid in both files: `TrackUID_1 < TrackUID_2`.

## Values to treat carefully

- **0028_vid** — median body length 159.2 px (range 15.8–1628.7); 1.9639% of speeds are above 20 body lengths/s; sex changes inside 33.0528% of track IDs.
- **0031_vid** — median body length 158.1 px (range 10.3–329.9); 1.1530% of speeds are above 20 body lengths/s; sex changes inside 30.7608% of track IDs.
- Treat sex as a per-frame tracking label, not a fixed fish attribute.
- There are no within-episode gaps longer than 0.2 s; the upstream converter started a new episode at that boundary.

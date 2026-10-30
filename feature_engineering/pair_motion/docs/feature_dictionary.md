# Feature dictionary

`outputs/pair_motion/pair_features_frame.parquet` has one row for an ordered pair of fish
on one video frame. It combines the original pair measurements with the two
matching fish tracks. `TrackUID_1 < TrackUID_2` keeps each pair in one order.

The source data uses pixels and radians. New distances use body lengths (`_bl`)
so trials can be compared even when fish size differs. The scale for a row is
the mean of `body_length_px_1` and `body_length_px_2`.

## Core columns

| Columns | Meaning |
| --- | --- |
| `project_id`, `day_index`, `day_label`, `Time_s`, `FrameNum` | Video and frame identifiers copied from the pair table. |
| `TrackID_1`, `TrackID_2`, `TrackUID_1`, `TrackUID_2` | The two fish tracks in the pair. |
| `pair_episode_id`, `dt_frames`, `smooth_segment` | A continuous pair track, elapsed frames since the previous row, and the smoothing segment. |
| `sex_1`, `sex_2`, `pair_sex` | Per-frame labels copied from the pair table. Do not treat them as fixed fish sex; see `qc_summary.md`. |

## Distance and proximity

| Column or pattern | Unit | Meaning |
| --- | --- | --- |
| `centroid_distance_px` | pixels | Distance between fish midline centroids. |
| `dist_nose1_nose2_px`, `dist_nose1_tailtip2_px`, `dist_nose2_tailtip1_px`, `dist_nose1_spine4of2_px`, `dist_nose2_spine4of1_px` | pixels | Keypoint-to-keypoint distances from the source pair table. |
| Matching `_bl` columns | body lengths | Pixel distance divided by the mean body length of the two fish. |
| `approach_rate_bl_s` | body lengths/s | Change in `centroid_distance_bl` divided by elapsed time. Positive means the fish are moving apart. |
| `closing_speed_bl_s` | body lengths/s | `-approach_rate_bl_s`. Positive means the fish are moving closer. |
| `proximity` | boolean | `true` when `centroid_distance_bl <= 1.5`. This is a starting threshold, not a circling label. |

## Orientation

| Column | Unit | Meaning |
| --- | --- | --- |
| `orbital_rad` | radians | Direction from fish 1 to fish 2. |
| `rel_heading_rad` | radians | Difference between the fish headings. |
| `bearing_1_rad`, `bearing_2_rad` | radians | Angle between each fish's heading and the direction to the other fish. Zero means the fish points toward the other. |
| `facing_1`, `facing_2` | boolean | `true` when the absolute bearing is below `pi/4` (45 degrees). |
| `mutual_facing` | boolean | Both fish are facing each other. |

## Speed and smoothing

| Column or pattern | Unit | Meaning |
| --- | --- | --- |
| `body_length_px_1`, `body_length_px_2` | pixels | Per-fish body length from the track table. |
| `speed_bl_s_1`, `speed_bl_s_2` | body lengths/s | Per-fish speed from the track table. |
| `speed_mean_bl_s` | body lengths/s | Average speed of the two fish. |
| `speed_diff_bl_s` | body lengths/s | Fish 1 speed minus fish 2 speed. |
| `*_smooth` | same as source | Centered 15-frame (0.5 s) rolling median followed by mean. It never crosses a pair episode or a gap longer than 0.2 s. Bearings are unwrapped before smoothing, then wrapped back to `[-pi, pi]`. |

## Data-quality columns

| Column | Meaning |
| --- | --- |
| `qc_n_missing_kp_1`, `qc_n_missing_kp_2` | Number of missing keypoints in each fish record. |
| `qc_missing_nose_or_tail` | `true` if either fish has a missing nose or tail coordinate. |
| `is_track_start_1`, `is_track_end_1`, `is_track_start_2`, `is_track_end_2` | Start or end of a source fish track. |

## Angular movement

Rates are radians per second. Rates with an absolute value above 15 rad/s are
tracking artifacts: they are flagged and replaced by NaN in the `_clean` columns
(the raw columns are kept).

| Column | Meaning |
| --- | --- |
| `d_orbital_rad_s`, `d_heading_1_rad_s`, `d_heading_2_rad_s`, `bend_angle_rad_1`, `bend_angle_rad_2` | Source values from the pair and track parquets. |
| `d_orbital_rad_s_clean`, `d_heading_1_rad_s_clean`, `d_heading_2_rad_s_clean` | Same rates with `abs > 15` set to NaN. |
| `abs_d_orbital_rad_s` | Absolute cleaned orbital rate: how fast the line between the fish rotates. |
| `orbital_step_rad` | Wrapped change in `orbital_rad` since the previous row of the same segment. Safe across the +/- pi seam. |
| `orbital_rate_derived_rad_s` | `orbital_step_rad` divided by elapsed time. On these files it equals the provided `d_orbital_rad_s` (correlation 1.0), which confirms its definition. |
| `orbital_accel_rad_s2` | Change in the cleaned orbital rate per second. |
| `corotation` | Mean over the two fish of `sign(d_heading) * sign(d_orbital)`. +1 when both turn the way the pair rotates, -1 when against it. |
| `heading_rate_abs_mean_rad_s`, `bend_abs_mean_rad` | Mean absolute heading rate and bend angle of the two fish. |
| `rel_heading_sin/cos`, `bearing_1_sin/cos`, `bearing_2_sin/cos` | Angle encodings without the +/- pi seam. `orbital_rad` is not encoded because it depends on the arbitrary pair order. |
| `qc_bad_body_length` | A body length that is not positive or is outside 0.5x to 2x the video median. Flagged, not removed. |
| `qc_angular_rate_clipped` | Any of the three rates above exceeded 15 rad/s on this row. |

## Time-windowed features

Columns are named `w<seconds>s_<statistic>` for 0.5, 1, 2 and 4 s (`w0p5s_`,
`w1s_`, `w2s_`, `w4s_`). Windows are measured in frames, centered on the row by
default (`trailing` is available for causal use), and stop at pair-episode and
gap boundaries. Rows missing from a window count as "not close" in the fractions
and are left out of the means.

| Statistic | Meaning |
| --- | --- |
| `close_frac` | Fraction of window frames with `proximity` true. |
| `mutual_facing_frac` | Fraction of window frames with `mutual_facing` true. |
| `coverage` | Fraction of window frames that have a row for this pair. |
| `distance_bl_mean`, `speed_mean_bl_s_mean`, `closing_speed_bl_s_mean`, `bend_abs_mean` | Window means over available rows. |
| `abs_d_orbital_mean`, `d_orbital_std`, `corotation_mean` | Orbital rate size, variability and corotation over close rows. |
| `orbit_turns_net`, `orbit_turns_abs` | Signed and total orbital rotation over close rows, in full turns (1 = 2 pi). |

Model-facing groups for ablations live in `feature_groups.py`. Sex, track ids and
pixel columns are never model features.

Color and behavior labels are not present in the provided parquets.

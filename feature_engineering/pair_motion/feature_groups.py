"""Model-facing feature groups, used for ablations and feature reports.

Raw pixel columns, ids, sex and QC flags are intentionally not model features.
"""

from __future__ import annotations

from . import angular_temporal as at

DISTANCE_PROXIMITY = [
    "centroid_distance_bl",
    "dist_nose1_nose2_bl",
    "dist_nose1_tailtip2_bl",
    "dist_nose2_tailtip1_bl",
    "dist_nose1_spine4of2_bl",
    "dist_nose2_spine4of1_bl",
    "centroid_distance_bl_smooth",
    "proximity",
]
ORIENTATION_FACING = [
    "facing_1",
    "facing_2",
    "mutual_facing",
    "rel_heading_sin",
    "rel_heading_cos",
    "bearing_1_sin",
    "bearing_1_cos",
    "bearing_2_sin",
    "bearing_2_cos",
]
SPEED_CLOSING = [
    "speed_bl_s_1",
    "speed_bl_s_2",
    "speed_mean_bl_s",
    "speed_diff_bl_s",
    "approach_rate_bl_s",
    "closing_speed_bl_s",
    "speed_mean_bl_s_smooth",
]
ANGULAR_MOVEMENT = [
    "d_orbital_rad_s_clean",
    "abs_d_orbital_rad_s",
    "orbital_rate_derived_rad_s",
    "orbital_accel_rad_s2",
    "d_heading_1_rad_s_clean",
    "d_heading_2_rad_s_clean",
    "heading_rate_abs_mean_rad_s",
    "corotation",
    "bend_abs_mean_rad",
]
TEMPORAL_CONSISTENCY = at.window_columns()

FEATURE_GROUPS = {
    "distance_proximity": DISTANCE_PROXIMITY,
    "orientation_facing": ORIENTATION_FACING,
    "speed_closing": SPEED_CLOSING,
    "angular_movement": ANGULAR_MOVEMENT,
    "temporal_consistency": TEMPORAL_CONSISTENCY,
}
ALL_MODEL_FEATURES = [column for group in FEATURE_GROUPS.values() for column in group]

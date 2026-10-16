# Cichlid visualization and experiments

Tools for inspecting existing pose data, testing proximity-based candidate generation, and optionally training a scene-level DLC2Action MS-TCN model.

**Start with [USER_GUIDE.md](USER_GUIDE.md)** for installation and complete commands. Run all commands from the shared repository root. The examples reference annotation and prediction CSVs in the repository; research videos and Parquet inputs must be obtained separately through the team's approved data location.

| Workflow | Required inputs | Command |
| --- | --- | --- |
| Visual review | Matching video and individual pose Parquet; optional labels/predictions | `python -m visualization.review visualization/seed0031.json` |
| Proximity experiment | Individual pose Parquet and reviewed labels; video for clips | `python -m visualization.experiment visualization/hypothesis0031.json --clips 3` |
| Optional learning | Pair Parquets and positive/negative labels, with explicit splits | See training commands in USER_GUIDE.md |

## Package contents

| File | Responsibility |
| --- | --- |
| review.py | Adapt data, extract a short clip, and export a review bundle |
| viewer.html | Playback, pose overlays, geometry plots, comparison timelines, and notes |
| core.py | Shared pose and viewer-data utilities |
| experiment.py | Proximity-rule sweep, label coverage, review workload, and representative clips |
| training.py | Optional preparation, training, inference, and learned-review integration |
| training_backend/ | DLC2Action model wrapper, preprocessing, and scene feature aggregation |
| seed0031.json | Example review configuration |
| hypothesis0031.json | Example experiment configuration |
| training_seed.json | Explicit seed-video train/test configuration |
| demo.py / demo_training.py | Synthetic examples without research inputs |
| requirements.txt | Visualization and experiment dependencies |
| requirements-learning.txt | Additional optional training dependencies |
| test_*.py | Adapter and workflow checks |

## Data contract and interpretation

Review and proximity experiments expect individual pose columns FrameNum, TrackUID, X_center, Y_center, and keypoint coordinate columns such as Nose_x/Nose_y. Optional training expects the original pair-table distance, angle, and angular-rate fields; an individual pose file alone is insufficient for that training adapter.

Frame intervals default to zero-based inclusive bounds. Timestamp labels are interpreted as elapsed video time with an exclusive end and converted at configured FPS. Confirm these conventions with the source author. Config file paths are relative to the JSON config; command-line output paths are relative to the working directory.

Unannotated time is unknown, not automatically negative. Explicit assumed-background training is available for exploratory use and documented in USER_GUIDE.md. Reviewer comments are preserved rather than automatically applied. Fish IDs can switch; labels/sex and reflections require visual verification. Scene-level scores do not identify the responsible fish pair.

For custom interval columns, set start/end in each track configuration; set end_exclusive=true when appropriate. Use kind=prediction for detector intervals. Add verified=true only when deliberately treating the selected source as accepted human labels. A frame-score source may be supplied through scores with path/frame/score/video column mappings. Interval mean_score or positive_fraction must not be substituted for frame probabilities.

## Test and review status

```bash
python -m unittest visualization.test_review visualization.test_experiment visualization.test_training -v
```

Thirteen tests and synthetic model training, inference, and learned-view export passed. Real-video alignment and browser playback require local verification. Synthetic examples establish software functionality, not research performance. New dataset schemas require an explicit adapter.

[REPOSITORY_REVIEW.md](REPOSITORY_REVIEW.md) records the branch snapshot reviewed for integration. [USER_GUIDE.md](USER_GUIDE.md) includes assumptions, output definitions, held-out evaluation guidance, and attribution.

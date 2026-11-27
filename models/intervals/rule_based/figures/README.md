# Data-quality examples (pose/tracking pipeline)

Each figure shows real frames from the original videos with the pipeline's keypoints drawn on top. Skeletons are
colored by TrackID (the same ID keeps its color across panels in one figure), a white ring marks the nose, and the
label shows `ID <TrackID> <sex label>`. The magenta line is the edge of the gravel floor.

| Figure | Problem | How common | What we do about it |
|---|---|---|---|
| `01_reflections.png` | A fish's mirror image in the tank glass is tracked as a second fish, with its own ID and sex label | 14% of detections (0031), 35% (0028) are off the floor, mostly reflections | Drop detections outside the floor outline (a proxy, see `06`) |
| `02_duplicate_detections.png` | Two IDs on one body; sometimes a real second fish is left untracked and its ID copies the first fish's keypoints | 3% of pair rows (0031), 6% (0028) closer than 40 px | Drop pairs < 40 px apart |
| `03_keypoints_jump_to_another_fish.png` | For one frame, a resting fish's skeleton is placed on a different, swimming fish while its box stays put | keypoints > 1 body length from their own box in 1% of rows (0031), 2% (0028) | Drop rows where skeleton and box disagree; clip orbit/turn rates above 15 rad/s |
| `04_id_break_during_circling.png` | Mid-circling, the same fish is handed a new TrackID | median track lasts 24 frames (< 1 s); each labeled circling window spans 2–12 IDs | Fine for frame-level detection; stitch tracks before any per-fish analysis |
| `05_sex_label_flips.png` | Same fish, same ID, one frame apart: the sex label changes | ~1/3 of tracks; 13–19 flips per 1,000 consecutive frames | Don't use per-frame sex as a feature; majority vote per episode at most |
| `06_cleaning_flaw_real_fish_off_floor.png` | Flaw in our own cleaning: a real fish swimming high near the back wall appears past the floor outline and is dropped as a reflection | not measured; no effect on circling detection (15/15 events with or without cleaning) | Possible fix: require a fish on the floor at the mirror position, moving with it |

Also found (not pictured): one DeepLabCut golden image, `labeled-data/0028_vid/img0335409.png`, is cropped to
923×843 instead of the video's 1296×972, so its hand labels are in a different coordinate system. It showed a false
100 px keypoint error and is excluded from the accuracy check; it should be re-extracted and re-labeled.

Frames referenced (video · FrameNum):
- Reflections: 0031 · 166991, 306945; 0028 · 690775, 553888
- Cleaning flaw (real fish flagged as reflection): 0031 · 363493, 363523 (tracks 20212, 20244)
- Duplicates: 0028 · 533806; 0031 · 338402, 264863
- Keypoint jump: 0031 · 169756–169758 (track 8076 → fish 8387)
- ID break: 0031 · 265650, 265656, 265662 (track 14373 → 14431, inside labeled window 265510–266227)
- Sex flips: 0031 · 362964–362965 (ID 20175); 0028 · 511394–511395 (ID 29481)

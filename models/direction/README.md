# Direction models

Given circling intervals, estimate **rotation direction** (CW/CCW on screen) and **number of turns/cycles**.
They do not decide *when* circling happens (see [`../intervals/`](../intervals/README.md)).

| Model | Method | Input → output |
|---|---|---|
| [`rule_based`](rule_based/README.md) | Rotation of the closest-pair line from pose tracking | any `outputs/predictions/<model>/` → `<model>_rule_labels/` with `Direction`, `Cycles` |
| [`optical-flow`](optical-flow/README.md) | A: Farneback flow on video; B: pose orbital angle; C: SAM 2 mask centroids | annotation / event CSVs → per-event direction, `num_turns` CSVs under `outputs/optical-flow/` |

Evaluation of `Direction`/`Cycles`: `evaluation/README.md` (rule labels are picked up as separate models).

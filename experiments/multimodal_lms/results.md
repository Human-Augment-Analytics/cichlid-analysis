# Five-clip local multimodal pilot

Model judgments, not verified labels or accuracy measurements.

Gemma 4 E4B (4-bit), Qwen3.5 4B (4-bit), Bonsai 27B (1-bit).

| Clip | Duration | Gemma | Qwen | Bonsai |
| --- | ---: | --- | --- | --- |
| `F1_613_266809-267100` | 9.73 s | yes; clockwise; 0.25 circles (medium) | yes; clockwise; 4.75 circles (high) | yes; clockwise; 1.25 circles (medium) |
| `F1_613_275040-275190` | 5.03 s | yes; clockwise; 0.25 circles (medium) | yes; clockwise; 1.5 circles (high) | yes; clockwise; 1.25 circles (medium) |
| `MC920_334703-334830` | 4.27 s | no; none; 0.0 circles (high) | yes; anticlockwise; 1.5 circles (high) | yes; clockwise; 1.25 circles (medium) |
| `MC920_437259-437437` | 5.97 s | yes; anticlockwise; 0.25 circles (medium) | yes; anticlockwise; 1.5 circles (high) | yes; clockwise; 1.25 circles (medium) |
| `MC920_713035-713330` | 9.87 s | yes; anticlockwise; 0.25 circles (medium) | yes; clockwise; 1.5 circles (high) | yes; clockwise; 1.25 circles (medium) |

All three models received identical timestamped JPEG frames for each clip (verified by SHA-256).
Sampling: approximately 4 fps, first and last frame included, 640 px maximum frame width; four timestamped frames per 2x2 contact sheet.
Direction is screen-space orbital motion; counts include partial revolutions rounded to quarter-turns.
Unknown circling and circle counts remain null; a null direction is normalized to uncertain.

Gemma's pipeline failed a time-reversal control on the first clip: it returned the same clockwise direction and identical explanation after the sampled frame order was reversed. Its direction/count outputs are unvalidated.

Qwen's last clip has internally inconsistent counting: its numeric circle_count is 1.5, while its evidence describes nine complete revolutions. The report preserves the numeric answer and the raw explanation; it does not reconcile them into a new label.

Bonsai 27B returned the same circling, direction, and count judgment for all five clips. This does not establish that those judgments are correct.

Bonsai passed a two-image color-order control in both orders. This verifies basic image transport/order, not motion understanding.

Counts and directions are model estimates. Confidence is self-reported and uncalibrated. No accuracy ranking can be inferred from these five comparisons.

## Existing annotation references

All five clips overlap manually annotated circling events. These labels were read separately for reporting and were never supplied to a model. Event boundaries differ from the clip boundaries above: event counts are contextual references, not exact clip-count targets. CW means clockwise; CCW means anticlockwise.

| Clip | Annotated event | Event frames | Behavior | Direction | Event circles |
| --- | --- | --- | --- | --- | ---: |
| `F1_613_266809-267100` | `0031_vid_evt_0029` | 266850–267030 | circling | CW | 2 |
| `F1_613_275040-275190` | `0031_vid_evt_0043` | 275040–275160 | circling | CCW | 1 |
| `MC920_334703-334830` | `0028_vid_evt_0002` | 334650–334830 | Circling | CCW | 2 |
| `MC920_437259-437437` | `0028_vid_evt_0048` | 437190–437460 | Circling | CCW | 2 |
| `MC920_713035-713330` | `0028_vid_evt_0083` | 712980–713310 | circling | CCW | 3.5 |

Sources: `ground_truth_annotations/finalized_annotations/circling_annotations_vid0028.csv` and `circling_annotations_vid0031.csv`.

## Model evidence

### F1_613_266809-267100 — Gemma 4 E4B

Pair: Two fish in a tank

The two fish maintain a relatively consistent distance and appear to be slowly orbiting each other in a clockwise direction. The motion is subtle, suggesting the beginning of a sustained orbit.

### F1_613_266809-267100 — Qwen3.5 4B

Pair: A pair of small, dark-colored fish swimming in a circular formation within a gravel-bottomed tank.

The fish maintain a consistent circular orbit around each other for the entire duration of the clip. The relative vector between the two fish rotates in a clockwise direction, completing approximately 4.75 revolutions.

### F1_613_266809-267100 — Bonsai 27B

Pair: Two small fish swimming in a sandy tank, maintaining a consistent relative position while rotating around each other.

The fish maintain a stable distance and form a rotating pair. The vector between them rotates clockwise over the observed period, completing approximately one full circle plus a quarter turn. Motion is consistent and visible throughout.

### F1_613_275040-275190 — Gemma 4 E4B

Pair: Two fish in a tank

The fish maintain a relatively consistent distance and appear to be slowly orbiting each other in a clockwise direction. The motion is subtle, suggesting the start of a slow orbit.

### F1_613_275040-275190 — Qwen3.5 4B

Pair: Two small, slender fish swimming in a gravel-bottomed aquarium

The fish maintain a consistent separation distance while rotating around a central point. The vector between them rotates in a clockwise direction, completing approximately 1.5 revolutions over the 5-second clip.

### F1_613_275040-275190 — Bonsai 27B

Pair: Two small fish swimming in a sandy tank, maintaining a consistent relative position while rotating around each other.

The fish maintain a stable distance and form a rotating pair. The vector between them rotates clockwise over the observed time, completing approximately one full revolution plus a quarter turn. Motion is consistent and visible throughout.

### MC920_334703-334830 — Gemma 4 E4B

Pair: Single fish in a tank

The video shows a single fish swimming within the confines of a tank. There is no second fish present to establish relative motion or circling.

### MC920_334703-334830 — Qwen3.5 4B

Pair: Two fish in a gravel tank, one dark and one lighter, swimming in close proximity.

The fish maintain a relatively constant distance from each other, forming a loose pair. Their relative position vector rotates anticlockwise. From t=0.000s to t=4.233s, the pair completes approximately 1.5 revolutions (a full circle plus a half-circle), with the motion being consistent and sustained rather than a single turn or chase.

### MC920_334703-334830 — Bonsai 27B

Pair: Two small fish swimming in a rectangular tank with gravel substrate, maintaining a consistent relative position while rotating around each other.

The fish maintain a stable distance and angular separation, tracing a clockwise orbital path around their common center of mass. The motion is consistent across multiple frames (t=0.000s to t=4.233s), with the relative position vector completing approximately 1 full revolution plus an additional quarter turn. The circular motion is sustained and does not reverse direction.

### MC920_437259-437437 — Gemma 4 E4B

Pair: Two fish in a tank

The two fish maintain a relatively consistent distance and appear to be slowly orbiting each other in an anticlockwise direction. The motion is slow, suggesting a small fraction of a revolution has occurred across the observed time span.

### MC920_437259-437437 — Qwen3.5 4B

Pair: Two fish in the top-left quadrant of the tank

The two fish in the top-left quadrant exhibit sustained orbital motion. From t=0.000s to t=5.933s, they maintain a consistent distance and rotate around each other in an anticlockwise direction. The relative vector between them completes approximately 1.5 full revolutions (360 degrees + 180 degrees) over the observed period.

### MC920_437259-437437 — Bonsai 27B

Pair: Two small fish swimming in a rectangular tank with gravel substrate, maintaining a consistent relative position while rotating around each other.

The fish maintain a stable distance and angular separation, tracing a clear clockwise orbital path around their common center of mass. The motion is sustained across multiple frames, though some occlusion and partial visibility at certain timestamps introduce moderate uncertainty in precise revolution counting.

### MC920_713035-713330 — Gemma 4 E4B

Pair: Two fish in a tank

The two fish maintain a relatively consistent distance and appear to be slowly orbiting each other in an anticlockwise direction. The motion is slow, suggesting a small fraction of a revolution has occurred across the observed time span.

### MC920_713035-713330 — Qwen3.5 4B

Pair: Two eels in the top-left corner of the tank

The two eels in the top-left corner engage in a sustained orbital motion. From t=0.000s to t=1.000s, they complete approximately one full revolution around each other. From t=1.000s to t=2.000s, they perform a second revolution. From t=2.000s to t=3.000s, they complete a third revolution. From t=3.000s to t=4.000s, they complete a fourth revolution. From t=4.000s to t=5.000s, they complete a fifth revolution. From t=5.000s to t=6.000s, they complete a sixth revolution. From t=6.000s to t=7.000s, they complete a seventh revolution. From t=7.000s to t=8.000s, they complete an eighth revolution. From t=8.000s to t=9.000s, they complete a ninth revolution. The motion is consistently clockwise.

### MC920_713035-713330 — Bonsai 27B

Pair: Two small fish swimming in a circular pattern around each other near the center of the tank, with one fish consistently to the left and above the other in a clockwise rotation.

The fish maintain a consistent relative position, rotating around each other in a clockwise direction. The trajectory shows approximately one and a quarter full revolutions over the observed time period. Visibility is good, but some frames have slight blurring which reduces confidence.


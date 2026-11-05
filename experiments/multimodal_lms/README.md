# Local multimodal clip pilot

Runs local LM Studio vision models on the same five original MP4 clips without
burned-in labels from `data/clips`. This is a qualitative pilot, not a held-out accuracy
evaluation. It does not modify annotations or detector predictions.

The selected clips are:

| Clip | Duration |
| --- | ---: |
| `F1_613_266809-267100.mp4` | 9.73 s |
| `F1_613_275040-275190.mp4` | 5.03 s |
| `MC920_334703-334830.mp4` | 4.27 s |
| `MC920_437259-437437.mp4` | 5.97 s |
| `MC920_713035-713330.mp4` | 9.87 s |

Selection covers both source videos and short complete clips; it does not use
ground-truth direction or count to select examples. All selected clips are shorter
than 12 seconds, so the default 48-frame cap retains approximately 4 fps.

## Protocol

- Use `lms ls --json` and `/api/v1/models` to inventory installed models.
- Load one model at a time with parallelism 1. The pilot uses 16,384 context
  tokens for Qwen and Bonsai, and 8,192 for Gemma after a memory-related retry.
- Decode the entire selected clip, sample uniformly at approximately 4 fps,
  include the first and last frame, resize each frame to at most 640 px wide,
  and encode JPEG at quality 85.
- Pack four consecutive frames into each 2x2 contact sheet with timestamps;
  read top-left, top-right, bottom-left, bottom-right. Send sheets in order via
  LM Studio's native `/api/v1/chat` endpoint. This is sampled-frame analysis,
  not native MP4 input. Model image preprocessing may further resize sheets.
  Grouping frames bounds vision-encoder memory: Gemma exhausted GPU memory
  with individual frames. Qwen is rerun with sheets for identical inputs.
- Supply identical frames and prompt to each model, with temperature 0,
  reasoning off, and a 600-token response limit. No prior chat is retained.
- Withhold clip filenames, directory names, ground-truth labels, pose estimates,
  and prediction labels from the model.
- Ask for circling, screen-space direction, circle count, confidence, a visual
  description of the pair, and visible evidence. Preserve uncertainty as null.
- Count revolutions of the relative-position vector between the same two fish.
  A circle is 360 degrees; include partial revolutions to the nearest quarter
  turn. Do not sum both fish's separate paths. Counts are model estimates.

`clockwise` and `anticlockwise` describe the overhead camera image, not a
world-coordinate direction. `mixed` allows reversals; `uncertain` permits an
unresolved direction. No circling means direction `none` and count 0.

## Run

Start the local server and load the desired installed variant (consult `lms ls`
for model keys; load the concrete variant if several are installed):

```sh
lms server start --bind 127.0.0.1 --port 1234
lms load qwen/qwen3.5-4b --context-length 16384 --parallel 1 -y
uv run python -m experiments.multimodal_lms.run \
  --model qwen/qwen3.5-4b \
  --contact-sheets \
  --clips F1_613_266809-267100.mp4 F1_613_275040-275190.mp4 \
    MC920_334703-334830.mp4 MC920_437259-437437.mp4 \
    MC920_713035-713330.mp4 \
  --output-dir outputs/experiments/multimodal_lms/qwen
```

Repeat with Gemma and Bonsai after unloading the previous model. Choose the
4-bit Gemma and 1-bit Bonsai variants to fit the local machine. The default
`--context-length 0` reuses the loaded model configuration; nonzero values ask the
server to reload and can cause a temporary memory peak. Model downloads and
inference stay local once weights are downloaded. No cloud inference is used.

Each output directory contains `config.json`, `available_models.json`, per-clip
JSON with frame indices, source metadata, input SHA-256, raw server response,
timings, and validated answer, plus `results.csv`. Successful matching requests
resume from cache. Invalid or failed responses remain errors, not negative labels.
Output files are under the existing ignored `outputs/experiments/` directory.
Generate the checked-in summary after all three runs:

```sh
uv run python -m experiments.multimodal_lms.report
```

Score the saved judgments against overlapping annotation references:

```sh
uv run python -m experiments.multimodal_lms.score
```

This writes `scores.csv`, `score_summary.csv`, `scores.md`, and `scores.html`.
Detection is scored as positive-clip recall; direction as agreement with the
overlapping event label. Count differences are provisional because all five
clip intervals differ from their reference event intervals. Exact clip-count
errors require annotations for the exact analyzed bounds.

The tested weights are `lmstudio-community/Qwen3.5-4B-MLX-4bit`,
`lmstudio-community/gemma-4-E4B-it-MLX-4bit`, and
`prism-ml/Bonsai-27B-mlx-1bit`. This is not a test of the originally installed
8-bit Gemma or 2-bit Bonsai variants, which exceeded local memory limits.
The installed MLX runtime is 1.14.0 on a 16 GB Apple Silicon machine.
The CLI currently cannot select `@1bit` at load time. For this run, Bonsai's
1-bit directory was temporarily exposed under a separate local model key using
hard links to the downloaded files, with no weight duplication or changes.
Its recorded inference identifier is `local-cichlid/bonsai-27b-mlx`.
The temporary alias's chat template was prefixed with `enable_thinking=false`
and set explicitly in copies of its tokenizer and processor configuration,
because a non-Hub model alias does not expose the native API's reasoning setting.
Its requests use `--omit-reasoning-parameter`; the original downloaded template,
configuration, and weights are unchanged. A short control verified a final
response with zero reasoning tokens before rerunning the clips. All models use
thinking disabled. The temporary alias is removed after the pilot.
For a later run, choose the 1-bit variant in LM Studio's model loader and pass
the loaded identifier shown by `lms ps` to `--model`.

For longer clips, increase `--max-frames` and the loaded context, or analyze
shorter intervals: the cap otherwise reduces the effective sampling rate,
which is recorded in the output. Sparse sampling can miss turns or reversals.

## Validation

```sh
uv run python -m unittest experiments.multimodal_lms.test_run
```

Tests check chronological sampling, inclusion of endpoints, preservation of
unknown labels, and rejection of inconsistent or truncated responses. Agreement
between models is not accuracy. Direction and circle-count accuracy require
independent human labels; the model's confidence is not calibrated.
The summary also includes overlapping human-labeled event references from the
finalized CSVs. Those event boundaries differ from the analyzed clip bounds,
so their circle counts must not be treated as exact clip-level targets.
All five overlap positive circling annotations, so this sample cannot estimate
false-positive performance.

An additional control reverses the sampled images of the first clip, retaining
ascending timestamps. Gemma returned the same clockwise direction and identical
explanation for forward and reversed motion. Treat its direction/count judgments
as unvalidated; this control failed to demonstrate sensitivity to time order.
The raw control response is `gemma/reversed_control.json` in the output directory.
Bonsai also passed a two-image color-order control in both orders (red/blue,
then blue/red). This checks basic transport and ordering, not fish-motion
understanding. The raw responses are in `bonsai/image_order_control.json`.

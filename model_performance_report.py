# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "imageio-ffmpeg==0.6.0",
#     "marimo==0.24.0",
#     "matplotlib==3.11.2",
#     "opencv-python-headless==5.0.0.93",
#     "polars==1.44.1",
# ]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")

with app.setup(hide_code=True):
    from colorsys import hsv_to_rgb
    from io import BytesIO
    from math import isfinite
    from pathlib import Path
    from urllib.request import urlretrieve

    import cv2
    import imageio_ffmpeg
    import marimo as mo
    import polars as pl
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch
    from matplotlib.ticker import StrMethodFormatter

    from evaluation.evaluate import (GROUND_TRUTH, evaluate, prediction_models, read_intervals,
                                     read_predictions, summary_view)
    from models.logistic_regression.features import KEYPOINTS
    from models.logistic_regression.train import VIDEO_FRAMES

    ROOT = Path(__file__).resolve().parent
    DATA_DIR = ROOT / "data"
    CLIPS_DIR = ROOT / "outputs" / "clips"
    FPS = 30


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Model performance report

    Evaluates every model with predictions in `outputs/predictions/<model_name>/<video_id>.csv`
    against the merged ground-truth spans, using the metrics from `evaluation/evaluate.py`.
    Regenerate a model's predictions with:

    | Model | Command |
    |-|-|
    | `logistic_regression` | `uv run python -m models.logistic_regression.predict` |
    | `rule_based` | `uv run python models/rule_based/detector.py` |

    The logistic regression's predictions include its training frames, and the rule-based
    detector was tuned on video 0031, so neither is a fully held-out evaluation.

    ## Metrics
    ### Mean over Frames (MoF)

    $$MoF = \frac{\text{\# of correct frames}}{\text{\# of all frames}}$$

    ### Edit Score
    Based on the Levenshtein or edit distance and tallies the minimum number of
    insertions, deletions, and replacement operations required
    to convert one segment sequence into another.

    $$\text{Edit Score} = \big ( 1 - \frac{e[|X|, |Y |]}{ max(|X|, |Y |) } \big ) \cdot 100 $$
    """)
    return


@app.cell
def _():
    # Download the source videos (~31 GB each, for clip previews) and the pose and
    # pair tables when missing.
    _downloads = {
        "0028_vid.mp4": "https://www.dropbox.com/scl/fo/zdkqd0n1tt6i9hf9wyta9/AMkLWMbEcHxMGEXGPuxG1NI/0028_vid.mp4?rlkey=l0gupg6wkgdd0fq5dbycuqm8v&st=m42jq5db&dl=1",
        "0031_vid.mp4": "https://www.dropbox.com/scl/fo/zdkqd0n1tt6i9hf9wyta9/ADCA61tJU9SWYSnh9DNWkkk/0031_vid.mp4?rlkey=l0gupg6wkgdd0fq5dbycuqm8v&st=xmrzc3hs&dl=1",
        "0028_vid.parquet": "https://www.dropbox.com/scl/fo/zdkqd0n1tt6i9hf9wyta9/AIzGAhjimK-vvUN1IRetNrI/0028_vid.parquet?rlkey=l0gupg6wkgdd0fq5dbycuqm8v&st=zd96dl6g&dl=1",
        "0028_vid_pairs.parquet": "https://www.dropbox.com/scl/fo/zdkqd0n1tt6i9hf9wyta9/AL6msxoSPODIp3jfuXxWb6Q/0028_vid_pairs.parquet?rlkey=l0gupg6wkgdd0fq5dbycuqm8v&st=nb7wminh&dl=1",
        "0031_vid_pairs.parquet": "https://www.dropbox.com/scl/fo/zdkqd0n1tt6i9hf9wyta9/AOLNU5Q-ydFtOxPr3Bppvss/0031_vid_pairs.parquet?rlkey=l0gupg6wkgdd0fq5dbycuqm8v&st=pkqm83so&dl=1",
        "0031_vid.parquet": "https://www.dropbox.com/scl/fo/zdkqd0n1tt6i9hf9wyta9/ACNtNV3xM1k06lXgc4zpB0o/0031_vid.parquet?rlkey=l0gupg6wkgdd0fq5dbycuqm8v&st=nox9c3q0&dl=1",
    }

    data_files = [download(_url, DATA_DIR / _filename) for _filename, _url in _downloads.items()]
    return


@app.function(hide_code=True)
def download(url, destination):
    """Download to a .part file first so interrupted downloads are retried."""
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {destination.name}...")
    try:
        urlretrieve(url, partial)
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return destination


@app.cell(hide_code=True)
def _():
    models = prediction_models()
    mo.stop(not models, mo.md("No predictions found in `outputs/predictions/`."))
    gt = read_intervals(GROUND_TRUTH)
    predictions = {_name: read_predictions(_dir) for _name, _dir in models.items()}
    evaluations = {_name: evaluate(gt, _predicted, num_frames=VIDEO_FRAMES)
                   for _name, _predicted in predictions.items()}
    return evaluations, gt, models, predictions


@app.cell(hide_code=True)
def _(evaluations, predictions):
    _comparison = pl.DataFrame([
        {"model": _name, "predicted_spans": predictions[_name].height,
         **{_key: _overall[_key] for _key in ("mof", "edit_score", "f1@10", "f1@25", "f1@50",
                                             "frame_precision", "frame_recall", "frame_f1")}}
        for _name, (_results, _) in evaluations.items()
        for _overall in [_results.row(-1, named=True)]
    ])
    mo.vstack([
        mo.md(f"## Model comparison\n\nOverall metrics across both videos · ground truth "
              f"`{GROUND_TRUTH.relative_to(ROOT)}`"),
        mo.ui.table(_comparison, selection=None, pagination=False),
        mo.ui.tabs({_name: summary_view(_results, _warnings)
                    for _name, (_results, _warnings) in evaluations.items()}),
    ])
    return


@app.function(hide_code=True)
def track_color(track_id):
    """Return a stable, distinct BGR color for a TrackID."""
    rgb = hsv_to_rgb((int(track_id) * 0.61803398875) % 1, 0.75, 1.0)
    return tuple(round(channel * 255) for channel in reversed(rgb))


@app.function(hide_code=True)
def overlay_poses(frame, frame_poses, scale=1.0):
    """Draw every track's keypoints, colored by TrackID, on a BGR frame in place."""
    height, width = frame.shape[:2]
    for row in frame_poses.iter_rows(named=True):
        color = track_color(row["TrackID"])
        for name in KEYPOINTS:
            x, y = row[f"{name}_x"], row[f"{name}_y"]
            if x is None or y is None or not isfinite(x) or not isfinite(y):
                continue
            point = round(x * scale), round(y * scale)
            if 0 <= point[0] < width and 0 <= point[1] < height:
                cv2.circle(frame, point, 4, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(frame, point, 3, color, -1, cv2.LINE_AA)
    return frame


@app.function(hide_code=True)
def overlay_legend(frame, track_ids, labels):
    """Top-left status badges and a top-right TrackID legend, drawn in place."""
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.5
    for index, (text, color) in enumerate(labels):
        text_width = cv2.getTextSize(text, font, scale, 1)[0][0]
        top = 8 + index * 24
        cv2.rectangle(frame, (8, top), (20 + text_width, top + 20), color, -1)
        cv2.putText(frame, text, (14, top + 15), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
    if not track_ids:
        return frame
    names = [f"TrackID {int(track_id)}" for track_id in track_ids]
    box_width = max(cv2.getTextSize(name, font, scale, 1)[0][0] for name in names) + 40
    x, y = frame.shape[1] - box_width - 8, 8
    cv2.rectangle(frame, (x, y), (x + box_width, y + 22 * len(names) + 10), (24, 24, 24), -1)
    for index, (track_id, name) in enumerate(zip(track_ids, names)):
        baseline = y + 22 + index * 22
        cv2.circle(frame, (x + 14, baseline - 5), 5, track_color(track_id), -1, cv2.LINE_AA)
        cv2.putText(frame, name, (x + 26, baseline), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


@app.cell(hide_code=True)
def _(models):
    clip_model = mo.ui.dropdown(options=list(models), value=next(iter(models)), label="Model",
                                allow_select_none=False)
    mo.vstack([
        mo.md("""
        ## Clip previews

        Pose keypoints are colored by TrackID. Frames inside the predicted span
        and inside ground-truth circling are labelled in the top-left corner.
        Previews are cached in `outputs/clips/<model_name>/`.
        """),
        clip_model,
    ])
    return (clip_model,)


@app.cell(hide_code=True)
def _(clip_model, predictions):
    _predicted = predictions[clip_model.value]
    mo.stop(_predicted.is_empty(), mo.md("No predicted spans to preview."))
    _options = {
        f"{_video} · {_start:,}–{_end:,} · {(_end - _start + 1) / FPS:.1f} s": (_video, _start, _end)
        for _video, _start, _end in _predicted.iter_rows()
    }
    clip_selection = mo.ui.dropdown(options=_options, value=next(iter(_options)),
                                    label="Predicted span", allow_select_none=False)
    clip_padding = mo.ui.number(start=0, stop=30, step=0.25, value=1, label="Padding (s)")
    mo.hstack([clip_selection, clip_padding], justify="start")
    return clip_padding, clip_selection


@app.function(hide_code=True)
def render_clip(video_id, start, end, gt_spans, output_dir, padding=0, width=648):
    """Cache a pose-overlay preview of a predicted span plus padding frames.

    Each frame lists its visible TrackIDs and is marked when it falls inside
    the predicted span or a ground-truth span. Bitrate is bounded so long spans stay around 2 MB.
    """
    first = max(0, start - padding)
    last = min(VIDEO_FRAMES[video_id] - 1, end + padding)
    output = output_dir / f"{video_id}_{start}-{end}_pad{padding}.mp4"
    if output.exists():
        return output
    poses = (pl.scan_parquet(DATA_DIR / f"{video_id}.parquet")
             .filter(pl.col("FrameNum").is_between(first, last)).collect())
    groups = poses.partition_by("FrameNum", as_dict=True)
    empty = poses.head(0)
    capture = cv2.VideoCapture(str(DATA_DIR / f"{video_id}.mp4"))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open {video_id}.mp4")
    scale = width / capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    size = (width, round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale / 2) * 2)
    bitrate = max(1000, min(250_000, int(12_000_000 / ((last - first + 1) / FPS))))
    output_dir.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".part.mp4")
    writer = imageio_ffmpeg.write_frames(
        str(partial), size, fps=FPS, pix_fmt_in="bgr24", codec="libx264",
        quality=None, macro_block_size=1, ffmpeg_log_level="error",
        output_params=["-preset", "fast", "-b:v", str(bitrate), "-maxrate", str(bitrate),
                       "-bufsize", str(2 * bitrate), "-movflags", "+faststart"],
    )
    try:
        writer.send(None)
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)
        for frame_number in range(first, last + 1):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode {video_id} frame {frame_number}")
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            frame_poses = groups.get((frame_number,), empty)
            overlay_poses(frame, frame_poses, scale)
            labels = [(f"Frame {frame_number:,}", (40, 40, 40))]
            if start <= frame_number <= end:
                labels.append(("Predicted circling", (199, 84, 121)))
            if any(s <= frame_number <= e for s, e in gt_spans):
                labels.append(("Ground truth circling", (142, 155, 21)))
            track_ids = sorted(frame_poses["TrackID"].drop_nulls().to_list())
            writer.send(overlay_legend(frame, track_ids, labels))
        writer.close()
        partial.replace(output)
    finally:
        capture.release()
        writer.close()
        partial.unlink(missing_ok=True)
    return output


@app.cell(hide_code=True)
def _(clip_model, clip_padding, clip_selection, gt, predictions):
    _video, _start, _end = clip_selection.value
    _frame_count = VIDEO_FRAMES[_video]
    _padding = int(clip_padding.value * FPS)
    _preview_start, _preview_end = max(0, _start - _padding), min(_frame_count - 1, _end + _padding)
    _lanes = [
        ("Ground truth", gt, "#159b8e"),
        (clip_model.value, predictions[clip_model.value], "#7954c7"),
    ]
    _radius = max(3000, (_end - _start + 1) * 3)
    _figure = Figure(figsize=(14, 3.5), layout="constrained")
    _axes = _figure.subplots(2, 1)
    for _axis, _limits, _title in zip(
        _axes,
        ((0, _frame_count), (max(0, _start - _radius), min(_frame_count, _end + 1 + _radius))),
        ("Full video", "Around selected span"),
    ):
        for _y, (_label, _spans, _color) in zip((1, 0), _lanes):
            _axis.broken_barh([(0, _frame_count)], (_y - 0.3, 0.6), facecolors="#edf0f3")
            _axis.broken_barh(
                [(_s, _e - _s + 1) for _s, _e in
                 _spans.filter(pl.col("video_id") == _video).select("start", "end").iter_rows()],
                (_y - 0.3, 0.6), facecolors=_color,
            )
        _axis.axvspan(_preview_start, _preview_end + 1, color="#f59e0b", alpha=0.15)
        _axis.axvspan(_start, _end + 1, color="#e11d48", alpha=0.2)
        _axis.set(xlim=_limits, ylim=(-0.5, 1.5), yticks=[1, 0], yticklabels=[l for l, _, _ in _lanes],
                  xlabel="Frame number (zero-based)")
        _axis.set_title(f"{_video} · {_title}", loc="left")
        _axis.tick_params(axis="y", length=0)
        _axis.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        _axis.spines[["top", "right", "left"]].set_visible(False)
    _figure.legend(
        handles=[Patch(color=_color, label=_label) for _label, _, _color in _lanes]
                + [Patch(color="#f59e0b", alpha=0.4, label="preview padding"),
                   Patch(color="#e11d48", alpha=0.4, label="selected")],
        loc="outside upper right", ncol=4, frameon=False,
    )
    _buffer = BytesIO()
    _figure.savefig(_buffer, format="png", dpi=150)
    _figure.clear()
    mo.image(_buffer.getvalue(), width="100%",
             alt="Ground-truth and predicted circling spans with the selected span highlighted")
    return


@app.cell(hide_code=True)
def _(clip_model, clip_padding, clip_selection, gt):
    _video, _start, _end = clip_selection.value
    _gt_spans = gt.filter(pl.col("video_id") == _video).select("start", "end").rows()
    with mo.status.spinner(title="Rendering pose-overlay preview..."):
        _clip = render_clip(_video, _start, _end, _gt_spans, CLIPS_DIR / clip_model.value,
                            padding=int(clip_padding.value * FPS))
    mo.vstack([
        mo.video(str(_clip), width="80%"),
        mo.md(f"`{_clip.relative_to(ROOT)}`"),
    ])
    return


if __name__ == "__main__":
    app.run()

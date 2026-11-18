# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "pyarrow",
# ]
# ///

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium", app_title="Data cleaning")

with app.setup:
    import os
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd

    import circling_cleaning as cc

    DATA_DIR = cc.default_data_dir(cc.ROOT)           # $CIRCLING_DATA_DIR, <repo>/data or ~/Downloads
    FIG_DIR = Path(__file__).resolve().parent / "figures"
    FPS = 30


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Cleaning the tracking data before detecting behavior

    **Cichlid behavior analysis · data-cleaning step · applies to every detector and feature**

    The lab pipeline turns video into tables of fish positions and fish pairs. Before any behavior can be
    detected, we checked those tables against the original video and found **three tracking artifacts that
    create fake fish pairs**, plus two issues that cleaning can't fix but that limit what the data can answer.

    Every frame below is real footage with the pipeline's skeletons drawn on top. Each skeleton is colored by
    track ID, a white ring marks the nose, and the label shows `ID <track ID> <sex label>`. The magenta line is
    the edge of the gravel floor.

    The rules live in one shared file, `models/rule_based/circling_cleaning.py`, so every notebook and detector cleans the data the
    same way.
    """)
    return


@app.cell
def _():
    # Flag every pair row in both videos (reads the full tracking tables: about a minute).
    flagged = {_v: cc.load_flagged(DATA_DIR, _v, ["FrameNum", "TrackID_1", "TrackID_2", "centroid_distance_px"])
               for _v in cc.VIDEOS}
    return (flagged,)


@app.cell
def _(flagged):
    def pct(s):
        return f"{s.mean():.1%}"

    rates = {_v: {"fish": flagged[_v][0], "pairs": flagged[_v][1]} for _v in cc.VIDEOS}
    return pct, rates


@app.cell(hide_code=True)
def _(pct, rates):
    def section(fig, title, what, rule, stat):
        img = FIG_DIR / fig
        return mo.vstack([
            mo.md(f"## {title}\n\n{what}"),
            mo.image(str(img)) if img.exists() else mo.md(f"_`{fig}` not found in `models/rule_based/figures/`._"),
            mo.md(f"**How common:** {stat}  \n**Cleaning rule:** {rule}"),
        ])

    _r = {_v: rates[_v] for _v in cc.VIDEOS}
    mo.vstack([
        mo.md("# Artifacts we remove"),
        section("01_reflections.png", "1. Reflections tracked as fish",
                "The camera sees each fish's mirror image in the glass walls, and the tracker gives the reflection its "
                "own ID and even a sex label. A fish beside its reflection looks exactly like a tight pair, which is "
                "what circling looks like. A real pair and a mirror pair can be told apart by motion: a reflection "
                "always moves in the same direction as its fish along the wall.",
                "drop any detection whose center is outside the traced gravel floor (a proxy; see *Limits of the "
                "cleaning rules* below).",
                f"{pct(_r['0031']['fish'].reflection)} of detections in 0031, "
                f"{pct(_r['0028']['fish'].reflection)} in 0028 are off the floor (mostly reflections, plus some real "
                "fish near the walls)."),
        section("02_duplicate_detections.png", "2. One fish tracked twice",
                "Two IDs sit on the same body, so a single fish counts as a pair a few pixels apart. Sometimes a real "
                "second fish is nearby but untracked, and its ID borrows the first fish's skeleton.",
                f"drop pairs closer than {cc.DUP_PX} px.",
                f"{pct(_r['0031']['pairs'].duplicate)} of pair rows in 0031, "
                f"{pct(_r['0028']['pairs'].duplicate)} in 0028."),
        section("03_keypoints_jump_to_another_fish.png", "3. Keypoints jump onto another fish",
                "For a single frame, a fish's skeleton is placed on a different fish while its bounding box stays put. "
                "The fish never moved, but the numbers say it teleported, which produces impossible turning speeds.",
                "drop detections whose skeleton center is more than one body length from its bounding box; "
                "turning rates above 15 rad/s are also clipped downstream.",
                f"{pct(_r['0031']['fish'].keypoint_jump)} of detections in 0031, "
                f"{pct(_r['0028']['fish'].keypoint_jump)} in 0028."),
    ])
    return


@app.cell(hide_code=True)
def _(rates):
    _rows = []
    for _v in cc.VIDEOS:
        _p = rates[_v]["pairs"]
        _rows.append({"video": _v, "pair rows": f"{len(_p):,}",
                      "reflection": f"{_p.reflection.mean():.1%}", "duplicate": f"{_p.duplicate.mean():.1%}",
                      "keypoint jump": f"{_p.keypoint_jump.mean():.1%}",
                      "kept after cleaning": f"{_p.keep.mean():.1%}"})
    mo.vstack([
        mo.md(r"""
    ## How much gets removed

    Share of **pair rows** flagged (a pair is flagged if either fish is). The categories overlap, so they don't
    add up to the total removed. Most removed pairs involve a reflection: in 0028, more than half of all "pairs"
    are a fish and its mirror image.
    """),
        pd.DataFrame(_rows),
    ])
    return


@app.cell(hide_code=True)
def _(rates):
    LABELS = {
        "0031": [(265510, 266227), (266809, 267100), (267242, 267837), (269089, 271817),
                 (272241, 273018), (273550, 274787), (275040, 275190), (276439, 276530)],
        "0028": [(334703, 334830), (334985, 335118), (335409, 335502), (336153, 336285),
                 (436636, 436895), (437259, 437437), (713035, 713330)],
    }

    def closest_kept(p):
        c = p.sort_values(["FrameNum", "centroid_distance_px"]).drop_duplicates("FrameNum")
        return c.set_index("FrameNum").keep

    _rows = []
    for _v in cc.VIDEOS:
        _k = closest_kept(rates[_v]["pairs"])
        _inside = np.zeros(int(_k.index.max()) + 1, dtype=bool)
        for _a, _b in LABELS[_v]:
            _inside[_a:_b + 1] = True
        _kin = _k[_inside[_k.index.to_numpy()]]
        _kout = _k[~_inside[_k.index.to_numpy()]]
        _rows.append({"video": _v,
                      "closest pair removed, inside labeled circling": f"{(~_kin).mean():.1%}",
                      "closest pair removed, rest of video": f"{(~_kout).mean():.1%}"})
    mo.vstack([
        mo.md(r"""
    ## Does cleaning remove real circling?

    Very little. During the human-labeled circling events the closest pair is almost always two real fish, so only
    about 5% of those frames lose their closest pair to cleaning, versus roughly half of the frames in the rest of the
    video. Cleaning removes fake pairs while leaving the behavior we want to detect almost untouched.
    """),
        pd.DataFrame(_rows),
    ])
    return


@app.cell(hide_code=True)
def _():
    def show_fig(name):
        img = FIG_DIR / name
        return mo.image(str(img)) if img.exists() else mo.md(f"_`{name}` not found._")

    mo.vstack([
        mo.md(r"""
    # Limits of the cleaning rules

    The rules are simple proxies, so they make mistakes. The one we have seen: **a real fish past the floor outline
    gets dropped as a reflection.** The camera looks down at an angle, so a fish swimming high near the back wall can
    appear beyond the floor edge. The frames below show two real fish swimming in opposite directions along the wall,
    which a fish and its reflection never do, yet the upper one is flagged.

    How often this happens has not been measured. It barely matters for circling detection (the rule-based detector
    catches the same 15/15 labeled events with or without cleaning), but it would matter for per-fish counts or a
    model that relies on distance alone. **Possible fix:** only call an off-floor detection a reflection when a fish on
    the floor sits at its mirror position and moves with it.
    """),
        show_fig("06_cleaning_flaw_real_fish_off_floor.png"),
        mo.md(r"""
    # Known issues cleaning can't fix

    These don't create fake pairs, so they don't affect *when* circling is detected. They do limit *which fish* and
    *which sex* questions until they're solved.

    ## 4. The same fish gets a new ID

    Track IDs last under a second (median 24 frames), so every circling event is split across 2–12 IDs.
    **Needed before per-fish analysis:** track stitching (linking IDs that are clearly the same fish).
    """),
        show_fig("04_id_break_during_circling.png"),
        mo.md(r"""
    ## 5. The sex label flips

    The sex label is decided frame by frame and changes within about a third of all tracks.
    **Guidance:** don't use per-frame sex as a feature; at most take a majority vote over an episode, and validate the
    classifier against known fish.
    """),
        show_fig("05_sex_label_flips.png"),
        mo.md(r"""
    ## Also found

    * `qc_n_missing_kp` is 0 in every row, so missing keypoints appear to be filled in upstream rather than flagged.
      We should ask how.
    * One golden (hand-labeled) image, `labeled-data/0028_vid/img0335409.png`, is cropped to 923×843 instead of
      1296×972, so its labels don't line up with the video. It should be re-extracted and re-labeled.
    """),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Using the cleaned data

    * **In code:** `from models.rule_based import circling_cleaning as cc` and call `cc.clean_pairs(data_dir, "0031")`.
    * **As files:** run `uv run python -m models.rule_based.circling_cleaning` to write `0031_vid_pairs_clean.parquet` and
      `0028_vid_pairs_clean.parquet`.

    The rule-based circling detector (`models/rule_based/detector.py`) uses this same module.
    """)
    return


if __name__ == "__main__":
    app.run()

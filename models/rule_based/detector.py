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
app = marimo.App(width="medium", app_title="Rule-based circling detector")

with app.setup:
    import os
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    import circling_cleaning as cc

    FPS = 30
    # Chart styling (validated categorical slot 1 for detections; neutral ink for labels and thresholds)
    INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    DET, LABEL = "#2a78d6", "#52514e"
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "font.size": 10})


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Detecting circling with two simple rules

    **Cichlid behavior analysis · offline circling detector · rule-based baseline**

    A pair of fish is marked as **circling** when:

    1. **they stay close:** within about one body length (160 px) of each other, and
    2. **the line between them keeps rotating in one direction:** at least **¾ of a full turn within 4 seconds**
       to start an episode.

    No machine learning, no training data, nothing hidden: every decision can be traced back to these two numbers.
    """)
    return


@app.cell(hide_code=True)
def _():
    _img = Path(__file__).resolve().parent / "figures" / "circling_explainer.png"
    mo.vstack([mo.md("## What circling looks like"),
               mo.image(str(_img)) if _img.exists() else mo.md("_`models/rule_based/figures/circling_explainer.png` not found._")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The rule in full

    | Step | Setting | In plain words |
    |---|---|---|
    | Start from cleaned data | tracking artifacts removed | see `models/rule_based/data_cleaning.py` |
    | Pick the pair | closest two fish in each frame | circling is always the two nearest fish |
    | **Close** | distance < **160 px** | about one body length |
    | **Start** an episode | ≥ **0.75 turns** in a 4 s window | the pair has clearly started orbiting |
    | **Keep going** | ≥ **0.4 turns** in a 4 s window | a looser bar once started, so a brief slowdown doesn't end the episode |
    | Join pauses | gaps < **3 s** are bridged | the annotators never split events closer than 4.7 s apart |
    | Minimum length | ≥ **2 s** | ignore flickers |

    The first two numbers (160 px, ¾ turn) are the original rule. The "keep going" threshold and pause-joining
    were added this week, tuned on video 0031 only, and then tested on video 0028.
    """)
    return


@app.cell
def _():
    DATA_DIR = cc.default_data_dir(cc.ROOT)           # $CIRCLING_DATA_DIR, <repo>/data or ~/Downloads
    OUT_DIR = cc.ROOT / "outputs" / "predictions" / "rule_based"
    CACHE = DATA_DIR / "rule_detector_cache"          # kept next to the data, outside git
    VIDEOS = ["0031", "0028"]
    LABELS = {
        "0031": [(265510, 266227), (266809, 267100), (267242, 267837), (269089, 271817),
                 (272241, 273018), (273550, 274787), (275040, 275190), (276439, 276530)],
        "0028": [(334703, 334830), (334985, 335118), (335409, 335502), (336153, 336285),
                 (436636, 436895), (437259, 437437), (713035, 713330)],
    }
    REVIEWED = {"0031": [(265510, 276530)],
                "0028": [(334703, 336285), (436636, 437437), (713035, 713330)]}
    return CACHE, DATA_DIR, LABELS, OUT_DIR, REVIEWED, VIDEOS


@app.cell
def _(CACHE, DATA_DIR, VIDEOS):
    # Per-frame closest-pair distance and rotation rate, before and after cleaning (rules in circling_cleaning.py).
    # The first run reads the full tracking tables (~1 min); later runs load a small cache (seconds).
    def build_series(v):
        _, p = cc.load_flagged(DATA_DIR, v, ["FrameNum", "TrackID_1", "TrackID_2", "centroid_distance_px",
                                             "d_orbital_rad_s"])
        n = int(p.FrameNum.max()) + 1
        out = {}
        for tag, q in (("raw", p), ("clean", p[p.keep])):
            q = q.sort_values(["FrameNum", "centroid_distance_px"]).drop_duplicates("FrameNum").set_index("FrameNum")
            q = q.reindex(pd.RangeIndex(0, n))
            out[f"dist_{tag}"] = q.centroid_distance_px.to_numpy()
            out[f"rate_{tag}"] = q.d_orbital_rad_s.where(q.d_orbital_rad_s.abs() <= 15).fillna(0).to_numpy()
        return out


    def load_series(v):
        path = CACHE / f"series_{v}.npz"
        if path.exists():
            return dict(np.load(path))
        s = build_series(v)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **s)
        return s


    series = {_v: load_series(_v) for _v in VIDEOS}
    return (series,)


@app.cell
def _():
    def signals(dist, rate, dist_max=160, win_s=4, frac_close_min=0.6):
        win = int(win_s * FPS)
        close = np.nan_to_num(dist, nan=1e9) < dist_max
        turns = (pd.Series(rate * close / FPS).rolling(win, center=True, min_periods=win // 2).sum().abs()
                 .to_numpy() / (2 * np.pi))
        frac = pd.Series(close.astype(float)).rolling(win, center=True, min_periods=win // 2).mean().to_numpy()
        return turns, frac >= frac_close_min


    def detect(turns, mostly_close, min_turns=0.75, keep_turns=None, merge_s=0, gap_s=1, min_dur_s=2):
        flag = (turns >= min_turns) & mostly_close
        if keep_turns is not None:
            weak = (turns >= keep_turns) & mostly_close
            run = np.cumsum(np.r_[1, np.diff(weak.astype(int)) != 0]) * weak
            started = np.unique(run[flag & weak])
            flag = np.isin(run, started[started > 0])
        idx = np.flatnonzero(flag)
        segs = []
        if len(idx):
            for g in np.split(idx, np.flatnonzero(np.diff(idx) > gap_s * FPS) + 1):
                if g[-1] - g[0] >= min_dur_s * FPS:
                    segs.append([int(g[0]), int(g[-1])])
        if merge_s and segs:
            merged = [segs[0]]
            for s, e in segs[1:]:
                if s - merged[-1][1] <= merge_s * FPS:
                    merged[-1][1] = e
                else:
                    merged.append([s, e])
            segs = merged
        return [tuple(s) for s in segs]
    return detect, signals


@app.cell
def _(VIDEOS, detect, series, signals):
    sig = {}
    detections = {}
    for _v in VIDEOS:
        _s = series[_v]
        sig[("original", _v)] = signals(_s["dist_raw"], _s["rate_raw"])
        sig[("refined", _v)] = signals(_s["dist_clean"], _s["rate_clean"])
        detections[("original", _v)] = detect(*sig[("original", _v)])
        detections[("refined", _v)] = detect(*sig[("refined", _v)], keep_turns=0.4, merge_s=3)
    return detections, sig


@app.cell
def _(LABELS, REVIEWED, VIDEOS, detections, series):
    def frame_mask(n, ranges):
        m = np.zeros(n, dtype=bool)
        for _a, _b in ranges:
            m[_a:_b + 1] = True
        return m


    def evaluate(v, segs, tol=FPS, null_reps=300):
        n = len(series[v]["dist_raw"])
        det, lab = frame_mask(n, segs), frame_mask(n, LABELS[v])
        rev_neg = frame_mask(n, REVIEWED[v]) & ~lab
        per = []
        for a, b in LABELS[v]:
            hit = [(s, e) for s, e in segs if not (e < a - tol or s > b + tol)]
            per.append({"hit": bool(hit), "pieces": len(hit), "coverage": float(det[a:b + 1].mean()),
                        "start_err": (min(s for s, _ in hit) - a) / FPS if hit else np.nan,
                        "end_err": (max(e for _, e in hit) - b) / FPS if hit else np.nan})
        per = pd.DataFrame(per)
        rng = np.random.default_rng(0)
        null = [sum(any(not (e < a - tol or s > b + tol) for s, e in [((s + o) % n, (e + o) % n) for s, e in segs])
                    for a, b in LABELS[v]) for o in rng.integers(0, n, null_reps)]
        return {"caught": int(per.hit.sum()), "events": len(per), "null": float(np.mean(null)),
                "coverage": float(det[lab].mean()), "pieces": float(per.pieces[per.hit].mean()),
                "start_err": float(per.start_err.abs().median()), "end_err": float(per.end_err.abs().median()),
                "false_alarm_s": (det & rev_neg).sum() / FPS, "reviewed_neg_s": rev_neg.sum() / FPS,
                "episodes": len(segs)}, per


    scores = {k: evaluate(k[1], s) for k, s in detections.items()}
    return (scores,)


@app.cell(hide_code=True)
def _(scores):
    _r31, _r28 = scores[("refined", "0031")][0], scores[("refined", "0028")][0]
    _o31 = scores[("original", "0031")][0]
    _caught = _r31["caught"] + _r28["caught"]
    _events = _r31["events"] + _r28["events"]
    mo.vstack([
        mo.md("## Results at a glance (refined detector)"),
        mo.hstack([
            mo.stat(value=f"{_caught} / {_events}", label="labeled events caught",
                    caption=f"randomly placed detections catch {_r31['null'] + _r28['null']:.1f}"),
            mo.stat(value="1 episode", label="per labeled event",
                    caption=f"was {_o31['pieces']:.1f} fragments in 0031"),
            mo.stat(value=f"{min(_r31['coverage'], _r28['coverage']):.0%}–{max(_r31['coverage'], _r28['coverage']):.0%}",
                    label="of labeled circling time covered", caption=f"was {_o31['coverage']:.0%} in 0031"),
            mo.stat(value=f"≤ {max(_r31['start_err'], _r28['start_err'], _r31['end_err'], _r28['end_err']):.1f} s",
                    label="median start/end error", caption="versus the human labels"),
        ], widths="equal", gap=1),
        mo.md("Video 0031 was used to choose the refinements; **video 0028 is the held-out test** and shows the same "
              "result."),
    ])
    return


@app.cell(hide_code=True)
def _(LABELS):
    _opts = {f"{v} · event {i + 1} · {(b - a + 1) / FPS:.0f} s at {a / FPS / 3600:.2f} h": (v, a, b)
             for v in ("0031", "0028") for i, (a, b) in enumerate(LABELS[v])}
    event_pick = mo.ui.dropdown(options=_opts, value=list(_opts)[0], label="Labeled event")
    mo.vstack([mo.md(r"""
    ## How the detector decides, one event at a time

    Top: distance between the two closest fish (they must stay under 160 px). Middle: how far the line between
    them rotated in the surrounding 4 s (an episode **starts** above 0.75 turns and **continues** while above 0.4).
    Bottom: the human label next to both detectors. Pick any labeled event:
    """), event_pick])
    return (event_pick,)


@app.cell(hide_code=True)
def _(detections, event_pick, series, sig):
    _v, _a, _b = event_pick.value
    _lo, _hi = max(0, _a - 15 * FPS), _b + 15 * FPS
    _t = np.arange(_lo, _hi) / FPS - _a / FPS          # seconds relative to the label start
    _dist = series[_v]["dist_clean"][_lo:_hi]
    _turns = sig[("refined", _v)][0][_lo:_hi]

    _fig, (_ax1, _ax2, _ax3) = plt.subplots(3, 1, figsize=(11, 6.2), sharex=True,
                                            gridspec_kw={"height_ratios": [2, 2, 1.2]})
    for _ax in (_ax1, _ax2):
        _ax.axvspan(0, (_b - _a) / FPS, color=GRID, zorder=0)
        _ax.grid(axis="y", color=GRID, linewidth=0.8)
        for _sp in ("top", "right"):
            _ax.spines[_sp].set_visible(False)
    _ax1.plot(_t, _dist, color=DET, linewidth=1.6)
    _ax1.axhline(160, color=INK2, linestyle="--", linewidth=1.2)
    _ax1.text(_t[-1], 160, "  close: 160 px", va="center", ha="left", color=INK2)
    _ax1.set_ylim(0, max(400, np.nanpercentile(_dist, 95) if np.isfinite(_dist).any() else 400))
    _ax1.set_title("Distance between the two fish (px)", loc="left", color=INK, fontsize=11)
    _ax2.plot(_t, _turns, color=DET, linewidth=1.6)
    for _y, _txt in ((0.75, "  start: 0.75"), (0.4, "  keep going: 0.4")):
        _ax2.axhline(_y, color=INK2, linestyle="--", linewidth=1.2)
        _ax2.text(_t[-1], _y, _txt, va="center", ha="left", color=INK2)
    _ax2.set_ylim(0, max(1.2, np.nanmax(_turns) * 1.1 if np.isfinite(_turns).any() else 1.2))
    _ax2.set_title("Rotation of the line between them (turns per 4 s)", loc="left", color=INK, fontsize=11)

    _rows = [("human label", [(_a, _b)], LABEL),
             ("original", detections[("original", _v)], DET),
             ("refined", detections[("refined", _v)], DET)]
    for _i, (_name, _spans, _col) in enumerate(_rows):
        _y = len(_rows) - 1 - _i
        for _s, _e in _spans:
            if _e >= _lo and _s <= _hi:
                _ax3.barh(_y, (_e - _s) / FPS, left=(_s - _a) / FPS, height=0.6, color=_col,
                          edgecolor=SURFACE, linewidth=1)
    _ax3.set_yticks(range(len(_rows)))
    _ax3.set_yticklabels([r[0] for r in _rows][::-1])
    _ax3.tick_params(axis="y", length=0)
    for _sp in ("top", "right", "left"):
        _ax3.spines[_sp].set_visible(False)
    _ax3.set_xlim(_t[0], _t[-1])
    _ax3.set_xlabel("seconds from the start of the human label (gray band = labeled circling)")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(REVIEWED):
    _opts = {f"{v} · {a / FPS / 60:.1f}–{b / FPS / 60:.1f} min": (v, a, b) for v in ("0031", "0028")
             for a, b in REVIEWED[v]}
    stretch_pick = mo.ui.dropdown(options=_opts, value=list(_opts)[0], label="Reviewed stretch")
    mo.vstack([mo.md(r"""
    ## Before vs. after this week's refinement

    Each bar is one labeled event or one detected episode, over a stretch of video a person reviewed. The original
    rule caught every event but broke long ones into fragments; the refined rule reports each event as one episode.
    """), stretch_pick])
    return (stretch_pick,)


@app.cell(hide_code=True)
def _(LABELS, detections, stretch_pick):
    _v, _a, _b = stretch_pick.value
    _lo, _hi = _a - 20 * FPS, _b + 20 * FPS
    _rows = [("human labels", LABELS[_v], LABEL), ("original detector", detections[("original", _v)], DET),
             ("refined detector", detections[("refined", _v)], DET)]
    _fig, _ax = plt.subplots(figsize=(11, 2.4))
    for _i, (_name, _spans, _col) in enumerate(_rows):
        _y = len(_rows) - 1 - _i
        for _s, _e in _spans:
            if _e >= _lo and _s <= _hi:
                _ax.barh(_y, (_e - _s) / FPS / 60, left=_s / FPS / 60, height=0.55, color=_col,
                         edgecolor=SURFACE, linewidth=1)
    _ax.set_yticks(range(len(_rows)))
    _ax.set_yticklabels([r[0] for r in _rows][::-1])
    _ax.tick_params(axis="y", length=0)
    _ax.set_xlim(_lo / FPS / 60, _hi / FPS / 60)
    _ax.set_xlabel(f"minutes into video {_v}")
    _ax.grid(axis="x", color=GRID, linewidth=0.8)
    for _sp in ("top", "right", "left"):
        _ax.spines[_sp].set_visible(False)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(scores):
    def _row(k):
        s = scores[k][0]
        return {"events caught": f"{s['caught']}/{s['events']}",
                "caught by random placement": f"{s['null']:.1f}",
                "labeled time covered": f"{s['coverage']:.0%}",
                "pieces per event": f"{s['pieces']:.1f}",
                "median start error": f"{s['start_err']:.1f} s",
                "median end error": f"{s['end_err']:.1f} s",
                "false alarm in reviewed, unlabeled time": f"{s['false_alarm_s']:.1f} of {s['reviewed_neg_s']:.0f} s",
                "episodes in the whole 10 h video": s["episodes"]}
    _tbl = pd.DataFrame({"0031 original": _row(("original", "0031")), "0031 refined": _row(("refined", "0031")),
                         "0028 original": _row(("original", "0028")),
                         "0028 refined (held out)": _row(("refined", "0028"))})
    mo.vstack([mo.md("**Numbers behind the comparison**"), _tbl])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Limitations and next steps

    **What we can't claim yet**

    * **Precision over the full videos is unknown.** Only a few minutes of each 10-hour video are labeled. Spot
      checks of detections outside the labels mostly showed real circling, but that needs a proper count.
    * **15 labeled events from 2 videos** is a small sample; treat the settings as provisional.
    * **Frame-level only.** The detector says *when* circling happens, not *which* fish: track IDs break up in
      under a second and the sex label flips, so per-fish questions need track stitching first.

    **Next steps**

    1. **Grow the ground truth:** review detected episodes as clips and record yes/no, plus fish IDs, event IDs and
       which stretches were reviewed.
    2. **Measure precision** on that reviewed sample and re-check the thresholds on more events.
    3. **Standardize the files** (ground truth, features, detections) so every detector can be scored with the same
       script.
    4. **Real time:** switch the 4 s window to past-only so the same rule can run live.
    """)
    return


@app.cell(hide_code=True)
def _(LABELS, OUT_DIR, VIDEOS, detections):
    _tables = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for _v in VIDEOS:
        _rows = [{"VideoID": f"{_v}_vid", "EventID": f"{_v}-R{_i:03d}", "StartFrame": _s, "EndFrame": _e,
                  "start_s": round(_s / FPS, 2), "end_s": round(_e / FPS, 2),
                  "duration_s": round((_e - _s + 1) / FPS, 2), "detector_name": "rule_v2",
                  "overlaps_label": any(not (_e < a or _s > b) for a, b in LABELS[_v])}
                 for _i, (_s, _e) in enumerate(detections[("refined", _v)], start=1)]
        _table = pd.DataFrame(_rows, columns=["VideoID", "EventID", "StartFrame", "EndFrame", "start_s", "end_s",
                                              "duration_s", "detector_name", "overlaps_label"])
        _table.to_csv(OUT_DIR / f"{_v}_vid.csv", index=False)
        _tables.append(_table)
    _export = pd.concat(_tables, ignore_index=True)
    mo.accordion({f"Appendix: all {len(_export)} detected episodes (saved to "
                  f"outputs/predictions/rule_based/<video>.csv)": _export})
    return


if __name__ == "__main__":
    app.run()

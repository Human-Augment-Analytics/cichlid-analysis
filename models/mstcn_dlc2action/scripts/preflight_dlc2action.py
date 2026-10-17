
import argparse
import importlib
import json
import logging
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TRAIN = "python -m models.mstcn_dlc2action.train"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import dlc2action_circling as cli
from dlc_yolo_converter import dlc2action_bridge as br


class Report:
    def __init__(self):
        self.errors = 0
        self.warnings = 0

    def section(self, title):
        print(f"\n== {title} " + "=" * max(0, 66 - len(title)))

    def ok(self, msg):
        print(f"  [ OK ] {msg}")

    def info(self, msg):
        print(f"         {msg}")

    def warn(self, msg):
        self.warnings += 1
        print(f"  [WARN] {msg}")

    def fail(self, msg):
        self.errors += 1
        print(f"  [FAIL] {msg}")


# ---------------------------------------------------------------------------
# 0. finalized annotations (train.py)
# ---------------------------------------------------------------------------

def repo_train_module():
    """models/mstcn_dlc2action/train.py (constants, FINALIZED_ANNOTATIONS)."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    return importlib.import_module("models.mstcn_dlc2action.train")


def check_finalized_annotations(r, paths):
    r.section("0. finalized annotations")
    try:
        t = repo_train_module()
    except Exception as e:
        r.fail(f"cannot import models/mstcn_dlc2action/train.py: {e!r}")
        return
   
    source = os.getenv("MSTCN_FINALIZED_ANNOTATIONS") or t.FINALIZED_ANNOTATIONS
    try:
        spans = t.read_annotations(source)
    except SystemExit as e:
        r.fail(str(e))
        return
    n = sum(len(v) for v in spans.values())
    r.ok(f"FINALIZED_ANNOTATIONS = {source}: {n} intervals "
         f"({', '.join(f'{v}: {len(i)}' for v, i in spans.items())}); "
         "all checks below use this file")
    return spans


def convert_annotations(r, spans, meta, out_dir):
    out_dir = Path(out_dir)
    src = out_dir / "finalized_annotations.csv"
    pd.DataFrame([{"VideoID": v, "StartFrame": s, "EndFrame": e}
                  for v, iv in spans.items() for s, e in iv]).to_csv(src, index=False)
    try:
        br.ground_truth_to_annotations(src, out_dir, meta)
    except Exception as e:
        r.fail(f"FINALIZED_ANNOTATIONS cannot be converted to H5 rows: {e}")
        return None
    return out_dir


def check_environment(r, args):
    r.section("1. environment")
    r.ok(f"python {sys.version.split()[0]}")
    for mod, why in (("numpy", ""), ("pandas", ""),
                     ("tables", "needed by pandas.read_hdf (pip install tables)"),
                     ("pyarrow", "needed to read parquet"),
                     ("ruamel.yaml", "DLC2Action config files"),
                     ("torch", ""),
                     ("polars", "train.py / predict.py (pip install polars)"),
                     ("sklearn", "train.py split and metrics (pip install scikit-learn)"),
                     ("matplotlib", "train.py precision-recall plot"),
                     ("joblib", "imported by the logistic-regression code"),
                     ("cv2", "DINOv2 crops (pip install opencv-python)")):
        try:
            m = importlib.import_module(mod)
            r.ok(f"{mod} {getattr(m, '__version__', '')}")
        except Exception as e:
            r.fail(f"cannot import {mod}: {e} {why}")
    try:
        br.ensure_dlc2action_importable()
        from dlc2action.project import Project  # noqa: F401
        ver = br.dlc2action_version()
        if ver in br.SUPPORTED_DLC2ACTION:
            r.ok(f"dlc2action {ver}")
        else:
            r.warn(f"dlc2action {ver}: this pipeline was verified against "
                   f"{br.SUPPORTED_DLC2ACTION}; the csv-store patch and "
                   "prediction format may differ")
        br.install_dlc2action_patches()
        from dlc2action import options
        if getattr(options.annotation_stores["csv"], "_circling_patched", False):
            r.ok("interactive csv annotation store patch active")
        else:
            r.fail("csv store patch did not install")
    except Exception as e:
        r.fail(f"dlc2action does not import: {e!r}")
        return
    try:
        import torch
        dev = getattr(args, "device", "auto")
        if torch.cuda.is_available():
            r.ok(f"CUDA GPU: {torch.cuda.get_device_name(0)}")
        elif dev in ("auto", "cpu"):
            r.warn("no CUDA GPU visible: training will run on CPU and be "
                   "very slow")
        if dev.startswith("cuda") and not torch.cuda.is_available():
            r.fail(f"--device {dev} but CUDA is not available")
    except Exception:
        pass


def check_paths(r, paths):
    r.section("2. paths")
    r.info(f"project root : {paths.root}")
    r.info(f"workdir      : {paths.workdir}")
    r.info(f"data (H5+csv): {paths.data}")
    r.info(f"projects     : {paths.projects}")
    for k in cli.LEGACY_ENV:
        if os.getenv(k):
            r.warn(f".env sets legacy {k}={os.getenv(k)} — it is ignored; "
                   "everything lives under the workdir above")
    if not paths.data.is_dir():
        r.fail(f"{paths.data} does not exist — run {TRAIN} (it converts the data)")
        return False
    try:
        paths.make()
        t = paths.workdir / ".write_test"
        t.write_text("x", encoding="utf-8")
        t.unlink()
        r.ok("workdir is writable")
    except Exception as e:
        r.fail(f"cannot write to {paths.workdir}: {e}")
    free_gb = shutil.disk_usage(paths.workdir).free / 1e9
    (r.ok if free_gb > 30 else r.warn)(
        f"{free_gb:.0f} GB free (DLC2Action caches features per segment; "
        "allow tens of GB for hour-long videos)")
    return True


def check_pose(r, paths, args, ann_dir=None):
    r.section("3. pose data (H5)")
    try:
        meta = br.load_meta(paths.data)
    except FileNotFoundError as e:
        r.fail(str(e))
        return None
    videos = sorted(meta)
    r.ok(f"metadata for {len(videos)} video(s): {videos}")
    h5s = sorted(p.stem for p in paths.data.glob("*.h5"))
    for s in sorted(set(h5s) - set(videos)):
        r.fail(f"{s}.h5 is not in the metadata (stale file?) — delete it or "
               "delete it and re-run train; DLC2Action would load it")
    canvas = cli.parse_canvas(getattr(args, "canvas", None)) or [1280, 720]
    thr = getattr(args, "likelihood_threshold", 0.5)
    if thr <= 0:
        r.fail("likelihood threshold must be > 0: missing keypoints are "
               "written with likelihood 0 and DLC2Action fills NaN with 0, "
               "so with 0 they become real (0,0) points")
    pos = (br.read_positive_intervals(ann_dir, videos, args.behaviors.split(",")[0])
           if ann_dir else {v: [] for v in videos})
    for v in videos:
        m = meta[v]
        h5 = paths.data / f"{v}.h5"
        if not h5.exists():
            r.fail(f"{v}: {h5.name} missing")
            continue
        try:
            df = pd.read_hdf(h5)
        except Exception as e:
            r.fail(f"{v}: cannot read {h5.name}: {e}")
            continue
        names = list(df.columns.names)
        if names != ["scorer", "individuals", "bodyparts", "coords"]:
            r.fail(f"{v}: column levels {names}, DLC2Action needs "
                   "[scorer, individuals, bodyparts, coords]")
            continue
        df = df.droplevel("scorer", axis=1)          # as DLC2Action does
        inds = sorted(df.columns.get_level_values("individuals").unique())
        coords = set(df.columns.get_level_values("coords"))
        if inds != ["ind0", "ind1"]:
            r.fail(f"{v}: individuals {inds}; need exactly ['ind0','ind1'] "
                   "for the pair clip 'ind0+ind1' (re-convert with "
                   "--max-individuals 2)")
        if coords != {"x", "y", "likelihood"}:
            r.fail(f"{v}: coords {sorted(coords)}")
        if len(df) != m["n_frames"]:
            r.fail(f"{v}: {len(df)} rows but metadata says {m['n_frames']} "
                   "— delete the data folder and re-run train")
        if not df.index.equals(pd.RangeIndex(len(df))):
            r.fail(f"{v}: index must be 0..n-1")
        lik = df.xs("likelihood", axis=1, level="coords").to_numpy()
        if not np.isin(lik, [0.0, 1.0]).all():
            r.warn(f"{v}: likelihoods are not 0/1; threshold {thr} applies "
                   "to real confidences")
        x = df.xs("x", axis=1, level="coords").to_numpy()
        y = df.xs("y", axis=1, level="coords").to_numpy()
        xm, ym = np.nanmax(x), np.nanmax(y)
        if xm > canvas[0] * 1.02 or ym > canvas[1] * 1.02:
            r.fail(f"{v}: keypoints reach x={xm:.0f}, y={ym:.0f} but canvas "
                   f"is {canvas[0]}x{canvas[1]} — set --canvas to the video "
                   "resolution")
        elif xm < canvas[0] * 0.5 and ym < canvas[1] * 0.5:
            r.warn(f"{v}: keypoints only reach x={xm:.0f}, y={ym:.0f}; is "
                   f"the canvas {canvas[0]}x{canvas[1]} right?")
        n_bp = len(df.columns.get_level_values("bodyparts").unique())
        present = lik.reshape(len(df), 2, n_bp).max(axis=2) > 0
        both = present.all(axis=1)
        fps = m.get("fps")
        hours = len(df) / fps / 3600 if fps else float("nan")
        pair_name = ("male + partner" if m.get("slot_assignment") ==
                     "male-partner" else "both fish")
        r.ok(f"{v}: {len(df)} rows ({hours:.2f} h at {round(fps, 3) if fps else '?'} fps, "
             f"stride {m.get('stride', 1)}, original frames "
             f"{m.get('frame_min')}..{m.get('frame_max')}); {pair_name} "
             f"visible in {100 * both.mean():.1f}% of rows, male in "
             f"{100 * present[:, 0].mean():.1f}%")
        st = m.get("slot_stats") or {}
        if st:
            r.info(f"{v}: {st.get('n_male_tracks')} male / "
                   f"{st.get('n_female_tracks')} female tracks, "
                   f"{st.get('partner_switches')} partner switches")
            if st.get("n_male_tracks", 0) == 0:
                r.fail(f"{v}: no male track found — check --male-values")
            if st.get("frames_multiple_male_tracks", 0) > 0.05 * len(df):
                r.warn(f"{v}: >1 male-classified track in "
                       f"{st['frames_multiple_male_tracks']} frames")
        elif m.get("slot_assignment") != "male-partner":
            r.warn(f"{v}: converted with slot assignment "
                   f"'{m.get('slot_assignment')}'; for one male + many "
                   "females re-convert (convert-data, default male-partner)")
        if not fps:
            r.warn(f"{v}: fps unknown (no Time_s); pass --fps to "
                   "export-events/score")
        if m.get("n_dropped_detections"):
            r.warn(f"{v}: {m['n_dropped_detections']} detections dropped "
                   f"(up to {m.get('max_simultaneous_tracks')} simultaneous "
                   "tracks) — spurious tracks may steal a slot")
        for s, e in pos.get(v, []):
            f = both[s:e].mean() if e > s else 0
            if f < 0.5:
                r.warn(f"{v}: bout rows {s}-{e}: {pair_name} visible in only "
                       f"{100 * f:.0f}% of rows (pair features interpolated)")
    return meta



def check_annotations(r, paths, args, meta, ann_dir, for_training=False):
    r.section("4. annotations (FINALIZED_ANNOTATIONS)")
    behaviors = set(args.behaviors.split(","))
    videos = sorted(meta)
    total_pos = 0
    for v in videos:
        d = pd.read_csv(Path(ann_dir) / f"{v}.csv")
        d.columns = [c.strip().lower() for c in d.columns]
        if d.empty:
            r.fail(f"{v}: no circling intervals in FINALIZED_ANNOTATIONS")
            continue
        dpos = d[d["behavior"].isin(behaviors)]
        short = dpos[(dpos["end"] - dpos["start"]) < args.min_frames_action]
        if len(short):
            r.warn(f"{v}: {len(short)} interval(s) shorter than "
                   f"min_frames_action={args.min_frames_action} rows will be "
                   "IGNORED by DLC2Action (labeled other)")
        n = int((dpos["end"] - dpos["start"]).sum())
        total_pos += n
        r.ok(f"{v}: {len(dpos)} interval(s) after merging overlaps, {n} positive rows "
             f"({100 * n / meta[v]['n_frames']:.3f}% of the video)")
    if total_pos == 0:
        r.fail("no positive frames at all")
    if any(int(m.get("frame_min", 0)) > 0 for m in meta.values()):
        r.info("annotation frames are on the original video axis and were shifted by "
               "each video's first detected frame (frame_min); FrameNum in the parquet "
               "must be the real video frame number")

    # DLC2Action reads data/<video>.csv, which train writes from the same file
    missing = [v for v in videos if not (paths.data / f"{v}.csv").exists()]
    stale = [v for v in videos if v not in missing and not
             pd.read_csv(paths.data / f"{v}.csv").equals(pd.read_csv(Path(ann_dir) / f"{v}.csv"))]
    if not missing and not stale:
        r.ok("data/<video>.csv (DLC2Action's copy) matches FINALIZED_ANNOTATIONS")
    elif for_training:
        r.fail(f"data/<video>.csv missing or different from FINALIZED_ANNOTATIONS for "
               f"{missing + stale}: re-run {TRAIN}")
    else:
        r.info(f"data/<video>.csv {'not written yet' if missing else 'out of date'} for "
               f"{missing + stale}; {TRAIN} writes them from FINALIZED_ANNOTATIONS")


def check_label_alignment(r, paths, args, meta, ann_dir):
    r.section("5. label alignment (DLC2Action's own annotation store)")
    try:
        br.install_dlc2action_patches()
        from dlc2action import options
    except Exception as e:
        r.fail(f"skipped, dlc2action unavailable: {e}")
        return
    videos = sorted(meta)
    behaviors = sorted(args.behaviors.split(","))
    step = args.len_segment - args.overlap
    try:
        import contextlib, io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            store = options.annotation_stores["csv"](
                video_order=videos,
                min_frames={v: {"ind0": 0, "ind1": 0} for v in videos},
                max_frames={v: {"ind0": meta[v]["n_frames"] - 1,
                                "ind1": meta[v]["n_frames"] - 1}
                            for v in videos},
                exclusive=True, len_segment=args.len_segment,
                overlap=args.overlap, behaviors=set(behaviors),
                annotation_suffix={".csv"}, annotation_path=str(ann_dir),
                filter_annotated=True, filter_background=False,
                error_class=br.ERROR_CLASS,
                min_frames_action=args.min_frames_action,
                default_agent_name=br.PAIR_CLIP, fps=1, interactive=True)
    except Exception as e:
        r.fail(f"DLC2Action could not build the annotation store: {e!r}")
        return
    coords = store.original_coordinates
    clips = {c[0].split("---")[1] for c in coords}
    expected = sum(len(br.segment_spans(meta[v]["n_frames"], args.len_segment,
                                        args.overlap)) for v in videos)
    if clips != {br.PAIR_CLIP}:
        r.fail(f"annotation clips {sorted(clips)}; expected only "
               f"{br.PAIR_CLIP} (labels would be misaligned)")
    if len(store) != expected:
        r.fail(f"annotation store has {len(store)} segments, the input store "
               f"will have {expected}: labels would be misaligned")
    else:
        r.ok(f"{len(store)} pair segments, same as the input store will make")
    bdict = store.behaviors_dict()
    if bdict != {0: "other", **{i + 1: b for i, b in enumerate(behaviors)}}:
        r.warn(f"class dictionary {bdict}")
    else:
        r.ok(f"classes {bdict}")
    # rebuild full-length labels from segments and compare to the csvs
    data = store.data.numpy() if hasattr(store.data, "numpy") else store.data
    full = {v: np.full(meta[v]["n_frames"], -999) for v in videos}
    for (vid_clip, i), lab in zip(coords, data):
        v = vid_clip.split("---")[0]
        s = int(i) * step
        n = meta[v]["n_frames"]
        e = min(s + args.len_segment, n)
        full[v][s:e] = lab[:e - s]
    bad = 0
    for v in videos:
        exp = np.zeros(meta[v]["n_frames"], dtype=int)
        d = pd.read_csv(Path(ann_dir) / f"{v}.csv")
        for b, s, e in zip(d["behavior"], d["start"], d["end"]):
            if b in behaviors and e - s >= args.min_frames_action:
                exp[s:e] = behaviors.index(b) + 1
        for b, s, e in zip(d["behavior"], d["start"], d["end"]):
            if b == br.ERROR_CLASS:
                exp[s:e] = -100
        mism = int((full[v] != exp).sum())
        if mism:
            bad += 1
            r.fail(f"{v}: {mism} rows where DLC2Action's labels differ from "
                   "the csv")
        else:
            r.ok(f"{v}: DLC2Action labels == csv for all "
                 f"{meta[v]['n_frames']} rows "
                 f"({int((exp > 0).sum())} circling, rest other)")


def check_split(r, paths, args, meta, ann_dir, for_training=False):
    r.section("6. train / val / test split")
    pm = args.partition_method
    videos = sorted(meta)
    n_frames = {v: meta[v]["n_frames"] for v in videos}
    beh = args.behaviors.split(",")[0]
    pos = br.read_positive_intervals(ann_dir, videos, beh)
    posmask = {v: np.zeros(n_frames[v], bool) for v in videos}
    for v in videos:
        for s, e in pos[v]:
            posmask[v][s:e] = True

    if pm == "random":
        r.fail(f"partition 'random' splits whole videos; with {len(videos)} "
               "videos val/test get none. Use the repo_split written by train.")
        return
    if pm.startswith("time"):
        step = args.len_segment - args.overlap
        for v in videos:
            spans = br.segment_spans(n_frames[v], args.len_segment, args.overlap)
            nv, nt = int(args.val_frac * len(spans)), int(args.test_frac * len(spans))
            parts = {"val": spans[:nv], "test": spans[nv:nv + nt],
                     "train": spans[nv + nt:]}
            for k, sp in parts.items():
                cnt = sum(posmask[v][s:e].sum() for s, e in sp)
                (r.ok if cnt else r.warn)(
                    f"{v} {k}: {len(sp)} segments, ~{cnt} positive rows")
        r.warn("time splits ignore where the bouts are; use the repo_split written by train")
        return
    if pm.startswith("val-from-name"):
        r.info("val/test chosen by video-name prefix")
        return
    if pm != "file":
        r.warn(f"partition method {pm!r} not checked")
        return
    if not args.split:
        r.fail("--split is required with partition method 'file'")
        return
    split = paths.split_file(args.split)
    if not split.exists():
        (r.fail if for_training else r.info)(
            f"split file {split.name} not written yet — {TRAIN} writes it")
        return
    try:
        parsed = br.read_split_file(split)
    except Exception as e:
        r.fail(str(e))
        return
    side = Path(str(split) + ".json")
    val_is_test = False
    if side.exists():
        sd = json.loads(side.read_text(encoding="utf-8"))
        if (sd["len_segment"], sd["overlap"]) != (args.len_segment, args.overlap):
            r.fail(f"split was made with len_segment/overlap "
                   f"{sd['len_segment']}/{sd['overlap']} but training uses "
                   f"{args.len_segment}/{args.overlap}: segment ids would not "
                   "match. Re-run train, which rewrites the split.")
            return
        if sd.get("n_frames") and sd["n_frames"] != n_frames:
            r.fail("split was made for different H5 lengths (data re-converted"
                   "?) — re-run train, which rewrites the split")
            return
        val_is_test = bool(sd.get("val_is_test"))
    else:
        r.warn("no split sidecar .json; grid compatibility checked from ids")
    valid = {v: set(br.segment_id(v, s, e) for s, e in br.segment_spans(
        n_frames[v], args.len_segment, args.overlap)) for v in videos}
    all_valid = set().union(*valid.values())
    seen = {}
    cover = {k: {v: np.zeros(n_frames[v], bool) for v in videos}
             for k in parsed}
    for k, entries in parsed.items():
        badids = [x for x in entries if x not in all_valid]
        if badids:
            r.fail(f"{k}: {len(badids)} segment ids not on the DLC2Action "
                   f"grid / unknown video, e.g. {badids[:2]}")
        for x in entries:
            if x in seen and seen[x] != k and not (
                    val_is_test and {seen[x], k} == {"val", "test"}):
                r.fail(f"segment {x} is in both {seen[x]} and {k}")
                break
            seen[x] = k
            v, _, rng = x.split("---")
            if v in cover[k]:
                s, e = map(int, rng.split("-"))
                cover[k][v][s:e] = True
    for k in ("train", "val", "test"):
        nseg = len(parsed[k])
        npos = sum(int((cover[k][v] & posmask[v]).sum()) for v in videos)
        nb = sum(1 for v in videos for s, e in pos[v]
                 if cover[k][v][s:e].mean() >= 0.5)
        msg = f"{k:5s}: {nseg} segments, {npos} positive rows, {nb} bouts"
        if nseg == 0:
            r.fail(msg + " — empty subset (DLC2Action would fill it with "
                   "every unassigned segment)")
        elif npos == 0 and k == "train":
            r.fail(msg + " — nothing to learn from")
        elif npos == 0:
            r.warn(msg + " — no positives: F1 here is meaningless")
        else:
            r.ok(msg)
    leak = 0
    for v in videos:
        held = cover["val"][v] | cover["test"][v]
        leak += int((cover["train"][v] & held).sum())
    if leak:
        r.fail(f"{leak} rows are in train AND val/test segments (leakage)")
    else:
        r.ok("no row is shared between train and val/test")


def _walk(d, prefix=""):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from _walk(v, f"{prefix}{k}.")
    else:
        yield prefix.rstrip("."), d


def check_project(r, paths, args, for_training):
    r.section("7. DLC2Action project")
    try:
        params = cli.training_params(args, paths)
        r.ok("training parameters build: interactive pair clip "
             f"'{params['data']['default_agent_name']}', canvas "
             f"{params['data']['canvas_shape']}, len_segment "
             f"{params['general']['len_segment']}/overlap "
             f"{params['general']['overlap']}, {args.num_epochs} epochs")
    except Exception as e:
        r.fail(f"cannot build training parameters: {e}")
        return
    exp = {k: params.get(k, {}) for k in ("general", "model", "augmentations",
                                          "features")}
    r.ok("experiment: presets "
         f"{sorted(set(getattr(args, 'preset', None) or [])) or 'none'}"
         f"{', overrides ' + str(args.param) if getattr(args, 'param', None) else ''}; "
         f"pair features {'on' if params['data'].get('feature_suffix') else 'off'}; "
         f"SSL {sorted(exp['general'].get('ssl') or []) or 'off'}"
         f" (PG_in_FE={exp['model'].get('PG_in_FE')}); augmentations "
         f"{sorted(exp['augmentations'].get('augmentations') or [])}; "
         f"{len(exp['features'].get('keys') or [])} feature keys")
    if exp["general"].get("ssl") and not (exp["model"].get("PG_in_FE")
                                          or exp["model"].get("num_layers_S")):
        r.fail("SSL is requested but MS-TCN3 has no feature extractor: set "
               "model.PG_in_FE=true (preset ssl does) or model.num_layers_S>0")
    proj = paths.projects / args.project_name
    if not proj.exists():
        r.ok(f"project '{args.project_name}' will be created in {paths.projects}")
        return
    from ruamel.yaml import YAML
    cfg = proj / "config"
    set_keys = {f"{sec}.{k}" for sec, vals in params.items()
                for k in (vals if isinstance(vals, dict) else {})}
    for y in [cfg / f"{s}.yaml" for s in ("general", "data", "training",
                                           "losses", "metrics", "ssl")]:
        if not y.exists():
            continue
        try:
            d = YAML().load(y.read_text(encoding="utf-8")) or {}
        except Exception as e:
            r.fail(f"{y.name} unreadable: {e}")
            continue
        for key, val in _walk(dict(d), f"{y.stem}."):
            if isinstance(val, str) and val.strip() == "???" and \
                    key not in set_keys:
                r.fail(f"{key} is '???' in the project config and not set by "
                       "train (run dlc2action_fill or set it)")
    gen = YAML().load((cfg / "general.yaml").read_text(encoding="utf-8")) if \
        (cfg / "general.yaml").exists() else {}
    for k, want in (("data_type", "dlc_track"), ("annotation_type", "csv")):
        if gen and gen.get(k) not in (None, want):
            r.fail(f"existing project has {k}={gen.get(k)!r}, need {want!r}; "
                   "use a new --project-name")
    data_yaml = cfg / "data.yaml"
    if data_yaml.exists():
        d = YAML().load(data_yaml.read_text(encoding="utf-8")) or {}
        old = d.get("data_path")
        if old and Path(str(old)).resolve() != paths.data.resolve():
            r.info(f"project data_path {old} will be updated to {paths.data}")
    try:
        project = cli._project(paths, args.project_name)
        names = list(project._episodes().data.index)
        try:
            ep = cli.default_episode(args)   # same name `train` will use
        except SystemExit as e:
            r.fail(str(e))
            return
        if ep in names and not args.overwrite:
            r.fail(f"episode '{ep}' already exists; pass --overwrite or a new "
                   "--episode")
        else:
            r.ok(f"episode name '{ep}' is free" if ep not in names else
                 f"episode '{ep}' will be overwritten")
    except Exception as e:
        r.warn(f"could not open the project to check episodes: {e}")


def smoke_test(r, paths, args, meta, ann_dir, keep=False):
    r.section("8. smoke test (crop -> train 2 epochs -> predict -> score)")
    import contextlib
    import io
    beh = args.behaviors.split(",")[0]
    videos = sorted(meta)
    pos = br.read_positive_intervals(ann_dir, videos, beh)
    smoke = paths.workdir / "_preflight_smoke"
    if smoke.exists():
        shutil.rmtree(smoke)
    sdata = smoke / "data"
    sdata.mkdir(parents=True)
    L = args.len_segment
    pad = 12 * L
    smeta, crops = {}, {}
    for v in videos:
        if not pos[v]:
            continue
        clusters = br.cluster_bouts({v: pos[v]}, 5000)
        c = clusters[0]
        s0 = max(0, c["start"] - pad)
        e0 = min(meta[v]["n_frames"], c["end"] + pad)
        df = pd.read_hdf(paths.data / f"{v}.h5").iloc[s0:e0]
        df.index = range(len(df))
        df.to_hdf(sdata / f"{v}.h5", key="df_with_missing", mode="w")
        a = pd.read_csv(Path(ann_dir) / f"{v}.csv")
        a = a[(a["end"] > s0) & (a["start"] < e0)].copy()
        a["start"] = (a["start"] - s0).clip(lower=0)
        a["end"] = (a["end"] - s0).clip(upper=e0 - s0)
        a.to_csv(sdata / f"{v}.csv", index=False)
        fsuf = (br.feature_file_suffix(args.feature_file)
                if getattr(args, "feature_file", None)
                else br.PAIR_FEATURE_SUFFIX if getattr(args, "pair_features", False)
                else None)
        if fsuf:
            import pickle
            pf = paths.data / f"{v}{fsuf}"
            if not pf.exists():
                r.fail(f"{v}: {pf.name} missing — run "
                       + f"{TRAIN} (it builds the feature files)")
                return
            with open(pf, "rb") as fh:
                feats = pickle.load(fh)
            with open(sdata / pf.name, "wb") as fh:
                pickle.dump({k: np.asarray(x)[s0:e0] for k, x in feats.items()}, fh)
        smeta[v] = dict(meta[v], n_frames=len(df), frame_min=0)
        crops[v] = [(s, e) for s, e in zip(a.loc[a["behavior"] == beh, "start"],
                                           a.loc[a["behavior"] == beh, "end"])]
    if sum(len(b) for b in crops.values()) < 2:
        r.warn("smoke test needs >= 2 bouts; skipped")
        return
    (sdata / br.META_FILENAME).write_text(json.dumps(smeta), encoding="utf-8")
    vv = max(crops, key=lambda k: len(crops[k]))
    b = sorted(crops[vv])
    if len(b) >= 2:
        cut = (b[len(b) // 2 - 1][1] + b[len(b) // 2][0]) // 2
    else:
        cut = b[0][0] - L
    entries = {"train": [], "val": []}
    for v in smeta:
        for s, e in br.segment_spans(smeta[v]["n_frames"], L, args.overlap):
            sid = br.segment_id(v, s, e)
            if v == vv and s >= cut + L:
                entries["val"].append(sid)
            elif v == vv and e > cut - L:
                continue
            else:
                entries["train"].append(sid)
    split = smoke / "smoke_split.txt"
    with open(split, "w", encoding="utf-8") as f:
        f.write("Training videos:\n" + "\n".join(entries["train"]) + "\n")
        f.write("Validation videos:\n" + "\n".join(entries["val"]) + "\n")
        f.write("Test videos:\n" + "\n".join(entries["val"]) + "\n")

    params = br.default_project_parameters(
        data_dir=sdata, behaviors=args.behaviors.split(","),
        len_segment=L, overlap=args.overlap,
        min_frames_action=args.min_frames_action,
        canvas_shape=cli.parse_canvas(args.canvas),
        partition_method="file", split_path=split, num_epochs=2,
        batch_size=min(args.batch_size, 32), device=args.device,
        num_cpus=min(args.num_cpus, 4),
        likelihood_threshold=args.likelihood_threshold,
        model_save_epochs=1,
        pair_features=getattr(args, "pair_features", False),
        feature_file=getattr(args, "feature_file", None),
        experiment=br.experiment_parameters(getattr(args, "preset", None) or [],
                                            getattr(args, "param", None) or []))
    want_ssl = bool(params["general"].get("ssl"))
    log = io.StringIO()
    t0 = time.time()
    stage = "project creation"
    try:
        br.install_dlc2action_patches()
        from dlc2action.project import Project
        (smoke / "projects").mkdir()
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            project = Project("smoke", data_type="dlc_track",
                              annotation_type="csv", data_path=str(sdata),
                              annotation_path=str(sdata),
                              projects_path=str(smoke / "projects"))
            project.update_parameters(params)
            stage = "training (2 epochs)"
            t1 = time.time()
            project.run_episode("smoke")
            t_train = time.time() - t1
            stage = "evaluate"
            metrics = project.evaluate(["smoke"], mode="test")
            stage = "prediction"
            project.run_prediction("smoke_pred", episode_names=["smoke"],
                                   data_path=str(sdata))
        r.ok(f"trained 2 epochs on {len(entries['train'])} segments in "
             f"{t_train:.0f}s (includes feature extraction)")
        out = log.getvalue()
        if want_ssl:
            if "no SSL possible" in out:
                r.fail("SSL requested but DLC2Action built MS-TCN without a "
                       "feature extractor ('no SSL possible')")
            elif "as a feature extractor" in out:
                r.ok("SSL active: MS-TCN uses the prediction generator as a "
                     "feature extractor")
            else:
                r.warn("could not confirm SSL from DLC2Action's output")
        elif "no SSL possible" in out:
            r.info("no SSL in this variant (expected)")
        r.ok(f"evaluate() works: {metrics}")
        stage = "loading predictions"
        pdir = smoke / "projects" / "smoke" / "results" / "predictions" / "smoke_pred"
        probs = br.load_prediction_probs(pdir, beh)
        for v in smeta:
            if v not in probs:
                r.fail(f"no prediction for {v}")
            elif len(probs[v]) != smeta[v]["n_frames"]:
                r.fail(f"{v}: prediction length {len(probs[v])} != "
                       f"{smeta[v]['n_frames']} rows")
        r.ok(f"predictions for {sorted(probs)} with the right lengths "
             f"(class index from the pickle's class dict)")
        stage = "per-frame scores + logistic-regression span rule"
        t = repo_train_module()
        from models.logistic_regression.postprocess import find_positive_spans
        n_spans = 0
        for v in smeta:
            n_fr = smeta[v]["n_frames"] * int(smeta[v].get("stride", 1))
            fs = t.frame_scores(probs[v], smeta[v], n_fr)
            n_spans += find_positive_spans(fs, fps=30, threshold=0.5, fraction=0.9,
                                           seconds=7.5).height
        r.ok(f"predict.py path works ({n_spans} spans after 2 epochs — the "
             "number is meaningless here)")
        test_rows = br.split_coverage_masks(split, {v: smeta[v]["n_frames"] for v in smeta})["test"]
        tp = fp = fn = 0
        for v in smeta:
            m = test_rows[v]
            lab = np.zeros(smeta[v]["n_frames"], bool)
            for s_, e_ in crops[v]:
                lab[s_:e_] = True
            pr = probs[v][:len(lab)] > 0.5
            tp += int((pr & lab)[m].sum())
            fp += int((pr & ~lab)[m].sum())
            fn += int((~pr & lab)[m].sum())
        d2a_f1 = metrics.get("f1") if isinstance(metrics, dict) else None
        ours = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")
        if d2a_f1 is not None and ours is not None and ours == ours:
            if abs(float(d2a_f1) - ours) > 0.15:
                r.fail(f"scorer disagrees with DLC2Action (F1 {ours:.3f} vs "
                       f"{float(d2a_f1):.3f}): prediction rows are probably "
                       "read in the wrong class order")
            else:
                r.ok(f"scorer agrees with DLC2Action's evaluate() (frame F1 "
                     f"{ours:.3f} vs {float(d2a_f1):.3f})")
        full_split = paths.split_file(args.split) if args.split else None
        if full_split and full_split.exists():
            n_full = len(br.read_split_file(full_split)["train"])
            per_seg_epoch = t_train / 2 / max(1, len(entries["train"]))
            eta_h = per_seg_epoch * n_full * args.num_epochs / 3600
            r.info(f"rough ETA for the real run: {n_full} train segments x "
                   f"{args.num_epochs} epochs ~ {eta_h:.1f} h on this machine "
                   "(+ one-off feature extraction). Overestimates when a GPU "
                   "amortizes batches better.")
    except Exception as e:
        r.fail(f"smoke test failed during {stage}: {e!r}")
        print("\n--- DLC2Action output (last 40 lines) ---")
        print("\n".join(log.getvalue().splitlines()[-40:]))
        print("--- traceback ---")
        traceback.print_exc()
    finally:
        if not keep:
            shutil.rmtree(smoke, ignore_errors=True)
        r.info(f"smoke test took {time.time() - t0:.0f}s"
               + (f"; kept in {smoke}" if keep else ""))



def run_preflight(args, paths, smoke=False, for_training=False, keep=False):
    import tempfile
    r = Report()
    print("DLC2Action circling — preflight")
    spans = check_finalized_annotations(r, paths)
    check_environment(r, args)
    with tempfile.TemporaryDirectory() as tmp:
        if check_paths(r, paths):
            try:
                meta = br.load_meta(paths.data)
            except FileNotFoundError:
                meta = None
            ann_dir = convert_annotations(r, spans, meta, tmp) if spans and meta else None
            meta = check_pose(r, paths, args, ann_dir)
            if meta and ann_dir:
                check_annotations(r, paths, args, meta, ann_dir, for_training)
                if r.errors == 0:
                    check_label_alignment(r, paths, args, meta, ann_dir)
                else:
                    r.section("5. label alignment")
                    r.info("skipped until the errors above are fixed")
                check_split(r, paths, args, meta, ann_dir, for_training)
            elif meta:
                r.section("4-6. annotations, label alignment, split")
                r.info("skipped: FINALIZED_ANNOTATIONS is not usable (section 0)")
            if meta:
                check_project(r, paths, args, for_training)
                if smoke:
                    if r.errors or not ann_dir:
                        r.section("8. smoke test")
                        r.info("skipped: fix the errors above first")
                    else:
                        smoke_test(r, paths, args, meta, ann_dir, keep=keep)
    print(f"\n{'READY' if r.errors == 0 else 'NOT READY'}: "
          f"{r.errors} error(s), {r.warnings} warning(s)")
    return r.errors


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", default=None)
    cli.add_training_args(ap)
    ap.add_argument("--smoke", action="store_true",
                    help="also run a 2-epoch end-to-end test on cropped data")
    ap.add_argument("--keep-smoke", action="store_true")
    try:
        t = repo_train_module()
        ap.set_defaults(workdir=str(t.WORKDIR), split=t.SPLIT, overlap=t.OVERLAP,
                        len_segment=t.LEN_SEGMENT, canvas=t.CANVAS, project_name=t.PROJECT,
                        episode=t.NAME, feature_file="pairemb", overwrite=True)
    except Exception as e:      # reported again in section 0
        print(f"  (could not read train.py defaults: {e!r})")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="  %(message)s")
    paths = cli.Paths(args.workdir)
    return min(run_preflight(args, paths, smoke=args.smoke,
                             keep=args.keep_smoke), 255)


if __name__ == "__main__":
    sys.exit(main())
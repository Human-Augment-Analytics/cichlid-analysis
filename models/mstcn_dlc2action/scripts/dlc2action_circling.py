#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError: 
    pass

from dlc_yolo_converter.dlc2action_bridge import (
    META_FILENAME,
    default_project_parameters,
    export_dlc_track_h5,
    ground_truth_to_annotations,
    install_dlc2action_patches,
    load_meta,
    make_event_split,
    predictions_to_events,
    read_ground_truth,
    read_positive_intervals,
    cluster_bouts,
    resolve_video_names,
    score_predictions,
    export_pair_features,
    experiment_parameters,
    preset_tag,
    PAIR_FEATURE_SUFFIX,
    PRESETS,
    FEATURE_FILES,
)

logger = logging.getLogger("dlc2action_circling")

LEGACY_ENV = ("DLC2ACTION_DATA_DIR", "DLC2ACTION_ANN_DIR",
              "DLC2ACTION_PROJECTS_DIR", "DLC2ACTION_FPS")



class Paths:
    def __init__(self, workdir=None):
        w = (workdir or os.getenv("DLC2ACTION_WORKDIR")
             or REPO / "outputs" / "mstcn_dlc2action")
        self.root = ROOT
        self.workdir = Path(w).expanduser().resolve()
        self.data = self.workdir / "data"
        self.splits = self.workdir / "splits"
        self.projects = self.workdir / "projects"
        self.events = self.workdir / "events"
        self.runs = self.workdir / "runs"

    def make(self):
        for p in (self.data, self.splits, self.projects, self.events,
                  self.runs):
            p.mkdir(parents=True, exist_ok=True)

    @property
    def meta(self):
        return self.data / META_FILENAME

    def split_file(self, split):
        p = Path(split)
        if p.suffix == ".txt" or p.exists():
            return p.resolve()
        return self.splits / f"{split}.txt"

    def prediction_dir(self, project, prediction):
        p = Path(prediction)
        if p.exists():
            return p.resolve()
        return self.projects / project / "results" / "predictions" / prediction


def warn_legacy_env():
    set_ = [k for k in LEGACY_ENV if os.getenv(k)]
    if set_:
        logger.warning(f"ignoring legacy .env variable(s) {set_}: all data "
                       "now lives under DLC2ACTION_WORKDIR "
                       "(default <root>/dlc2action)")


def env_default(name, fallback):
    v = os.getenv(name)
    return v if v not in (None, "") else fallback


def parse_canvas(s):
    if not s:
        return None
    w, h = s.lower().split("x")
    return [int(w), int(h)]


def _parse_name_map(pairs):
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--name-map entries must be OLD=NEW, got {p!r}")
        old, new = p.split("=", 1)
        out[old.strip()] = new.strip()
    return out


def training_params(args, paths):
    split_path = None
    if args.partition_method == "file":
        split_path = paths.split_file(args.split)
    return default_project_parameters(
        data_dir=paths.data,
        behaviors=args.behaviors.split(","),
        len_segment=args.len_segment,
        overlap=args.overlap,
        min_frames_action=args.min_frames_action,
        canvas_shape=parse_canvas(args.canvas),
        partition_method=args.partition_method,
        split_path=split_path,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        device=args.device,
        num_cpus=args.num_cpus,
        likelihood_threshold=args.likelihood_threshold,
        pair_features=getattr(args, "pair_features", False),
        feature_file=getattr(args, "feature_file", None),
        experiment=experiment_parameters(getattr(args, "preset", None) or [],
                                         getattr(args, "param", None) or []),
    )


def cmd_convert_data(args):
    paths = Paths(args.workdir)
    paths.make()
    gt_videos = []
    if args.ground_truth:
        gt_videos = sorted(read_ground_truth(args.ground_truth)["video"].unique())
    stems = [Path(k).stem for k in args.keypoints]
    names = resolve_video_names(stems, gt_videos,
                                _parse_name_map(args.name_map))
    meta = json.loads(paths.meta.read_text(encoding="utf-8")) if paths.meta.exists() else {}
    for kp in args.keypoints:
        kp = Path(kp)
        video = names[kp.stem]
        if video != kp.stem:
            logger.info(f"{kp.name} -> video id '{video}'")
        meta[video] = export_dlc_track_h5(
            parquet_path=kp,
            out_h5=paths.data / f"{video}.h5",
            scorer=args.scorer,
            video_name=video,
            max_individuals=args.max_individuals,
            slot_assignment=args.slot_assignment,
            max_gap=args.max_gap,
            frame_offset=args.frame_offset,
            stride=args.stride,
            float32=not args.no_float32,
            compress=not args.no_compress,
            male_values=args.male_values,
            switch_factor=args.switch_factor,
        )
    paths.meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    stray = sorted(p.stem for p in paths.data.glob("*.h5") if p.stem not in meta)
    if stray:
        logger.warning(f"H5 files without metadata in {paths.data}: {stray} "
                       "(delete them or re-convert)")
    if gt_videos and set(gt_videos) - set(meta):
        logger.warning(f"ground-truth videos without H5: "
                       f"{sorted(set(gt_videos) - set(meta))}")
    print(f"\nH5 + metadata -> {paths.data}")
    print("Next: convert-ann (re-run it whenever you re-convert pose data).")


def default_episode(args):
    if args.episode:
        return args.episode
    if getattr(args, "param", None) and not getattr(args, "tag", None):
        raise SystemExit("--param changes the model: name the run with --tag "
                         "(or --episode) so it doesn't overwrite another variant")
    base = (Path(args.split).stem if args.partition_method == "file"
            and args.split else "episode")
    if getattr(args, "pair_features", False) and getattr(args, "feature_file", None):
        raise SystemExit("--pair-features and --feature-file are exclusive: the "
                         "feature file already contains the pair features")
    name = base + ("_pair" if getattr(args, "pair_features", False) else "")
    if getattr(args, "feature_file", None):
        name += f"_{args.feature_file}"
    name += preset_tag(getattr(args, "preset", None) or [])
    if getattr(args, "tag", None):
        name += f"_{args.tag}"
    return name


def cmd_convert_pairs(args):
    paths = Paths(args.workdir)
    meta = load_meta(paths.data)
    stems = [Path(k).stem for k in args.pairs]
    stripped = {s: s.replace("_pairs", "").replace("_pair", "") for s in stems}
    names = resolve_video_names(list(stripped.values()), sorted(meta),
                                _parse_name_map(args.name_map))
    for pp in args.pairs:
        pp = Path(pp)
        video = names[stripped[pp.stem]]
        logger.info(f"{pp.name} -> video id '{video}'")
        stats = export_pair_features(
            pp, meta[video], paths.data / f"{video}{PAIR_FEATURE_SUFFIX}")
        meta[video]["pair_stats"] = stats
    paths.meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    missing = sorted(v for v in meta
                     if not (paths.data / f"{v}{PAIR_FEATURE_SUFFIX}").exists())
    if missing:
        logger.warning(f"no pair features yet for {missing}: with "
                       "--pair-features DLC2Action would SKIP those videos")
    print(f"\nPair features -> {paths.data}")
    print("Next: preflight / train with --pair-features "
          "(episodes are named <split>_pair).")


def cmd_convert_ann(args):
    paths = Paths(args.workdir)
    meta = load_meta(paths.data)
    written = ground_truth_to_annotations(
        gt_csv=args.ground_truth,
        out_dir=paths.data,
        meta=meta,
        default_fps=args.fps,
        default_label=args.label,
        merge_existing=args.merge,
    )
    (paths.data / "_annotation_source.json").write_text(json.dumps({
        "ground_truth": str(Path(args.ground_truth).resolve()),
        "label": args.label, "fps": args.fps, "merged": bool(args.merge),
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {len(written)} annotation file(s) -> {paths.data}")


def cmd_make_split(args):
    paths = Paths(args.workdir)
    paths.make()
    if args.list_clusters:
        meta = load_meta(paths.data)
        pos = read_positive_intervals(paths.data, sorted(meta), args.behavior)
        clusters = cluster_bouts(pos, args.cluster_gap)
        print(f"{len(clusters)} bout cluster(s) (gap <= {args.cluster_gap} "
              "H5 rows):")
        for c in clusters:
            print(f"  {c['id']}: {c['video']} rows {c['start']}-{c['end']} "
                  f"bouts={len(c['bouts'])} positive_rows={c['pos_frames']}")
        return
    if args.scheme == "cluster" and args.fold == "all":
        meta = load_meta(paths.data)
        pos = read_positive_intervals(paths.data, sorted(meta), args.behavior)
        folds = range(len(cluster_bouts(pos, args.cluster_gap)))
    elif args.scheme == "crossvideo" and args.fold == "all":
        folds = range(len(load_meta(paths.data)))     
    else:
        folds = [int(args.fold)]
    for k in folds:
        if args.name and "{k}" in args.name:   
            name = args.name.format(k=k)
        else:
            name = args.name or (f"fold{k}" if args.scheme in ("cluster", "crossvideo")
                                 else "videosplit")
            if args.fold == "all":
                name = f"{args.name}{k}" if args.name else f"fold{k}"
        out = paths.splits / f"{name}.txt"
        s = make_event_split(
            data_dir=paths.data, out_path=out,
            len_segment=args.len_segment, overlap=args.overlap,
            scheme=args.scheme, fold=k, val_cluster=args.val_cluster,
            cluster_gap=args.cluster_gap, context=args.context,
            guard=args.guard,
            test_videos=args.test_videos or [],
            val_videos=args.val_videos or [],
            train_background_frac=args.train_background_frac,
            near_bout=args.near_bout, behavior=args.behavior,
            seed=args.seed, val_frac=args.val_frac_split)
        print(f"\nsplit '{name}' -> {out}")
        if s["scheme"] == "cluster":
            print(f"  test = cluster {s['fold']}, val = cluster "
                  f"{s['val_cluster']} (of {s['n_clusters']})")
        elif s["scheme"] == "crossvideo":
            print(f"  test = {s['test_videos']} (whole video); train = "
                  f"{s['train_videos']}; val = tail of {s['val_video']} "
                  f"({s['val_clusters']} of {s['val_clusters'] + s['train_clusters']} "
                  f"clusters, {100 * s['val_pos_frac']:.0f}% of training positives)")
        for sub in ("train", "val", "test"):
            c = s["coverage"][sub]
            print(f"  {sub:5s}: {c['segments']:6d} segments  "
                  f"{c['frames']:9d} rows  {c['pos_frames']:7d} positive rows  "
                  f"{c['bouts']:3d} bouts")
        print(f"  excluded (guard bands / background subsampling): "
              f"{s['n_excluded_segments']} segments")


def _project(paths, name, create=False):
    install_dlc2action_patches()
    from dlc2action.project import Project
    paths.projects.mkdir(parents=True, exist_ok=True)
    kw = {"projects_path": str(paths.projects)}
    if (paths.projects / name).exists():
        return Project(name, **kw)
    if not create:
        raise SystemExit(f"project '{name}' not found in {paths.projects}")
    return Project(name, data_type="dlc_track", annotation_type="csv",
                   data_path=str(paths.data), annotation_path=str(paths.data),
                   **kw)


def cmd_train(args):
    paths = Paths(args.workdir)
    paths.make()
    episode = default_episode(args)
    args.episode = episode
    if not args.skip_preflight:
        import preflight_dlc2action as pf
        n_err = pf.run_preflight(args, paths, smoke=False, for_training=True)
        if n_err:
            raise SystemExit(f"\npreflight found {n_err} error(s); not "
                             "training. Fix them or pass --skip-preflight.")
    params = training_params(args, paths)
    recorded = json.loads(json.dumps(params, default=sorted))
    project = _project(paths, args.project_name, create=True)
    project.update_parameters(params)

    run_record = {"episode": episode, "project": args.project_name,
                  "split": (str(paths.split_file(args.split))
                            if args.partition_method == "file" else None),
                  "partition_method": args.partition_method,
                  "presets": sorted(set(getattr(args, "preset", None) or [])),
                  "param_overrides": list(getattr(args, "param", None) or []),
                  "params": recorded}
    (paths.runs / f"{episode}.json").write_text(json.dumps(run_record,
                                                           indent=2), encoding="utf-8")
    if args.search:
        from dlc2action import options
        space = {k: v for k, v in options.model_hyperparameters["ms_tcn3"].items()
                 if k != "general/len_segment"}
        print(f"search space: {sorted(space)}")
        project.run_hyperparameter_search(
            search_name=args.search_name, search_space=space, metric="f1",
            n_trials=args.search_trials, best_n=3, direction="maximize",
            prune=True, force=args.overwrite)
        project.run_episode(episode, load_search=args.search_name,
                            force=args.overwrite)
    else:
        project.run_episode(episode, force=args.overwrite)
    try:
        metrics = project.evaluate([episode], mode="test")
        print(f"\nDLC2Action evaluation on the TEST subset (frame-level, "
              f"circling only, threshold 0.5): {metrics}")
    except Exception as e:
        logger.warning(f"evaluate() failed ({e}); the model is saved. Use "
                       "predict + score instead.")
    print(f"\nDone. Model + metrics under "
          f"{paths.projects / args.project_name / 'results'}")
    print(f"Next: predict --episode {episode}  then  score --prediction "
          f"{episode}_pred --split {args.split or '<split>'}")


def cmd_predict(args):
    paths = Paths(args.workdir)
    project = _project(paths, args.project_name)
    name = args.prediction_name or f"{args.episode}_pred"
    data = Path(args.data or paths.data).resolve()
    project.run_prediction(name, episode_names=[args.episode],
                           data_path=str(data), force=args.overwrite,
                           augment_n=args.augment_n)
    print(f"\nPrediction -> {paths.prediction_dir(args.project_name, name)}")


def cmd_export_events(args):
    paths = Paths(args.workdir)
    pred = paths.prediction_dir(args.project_name, args.prediction)
    meta_path = Path(args.meta) if args.meta else paths.meta
    df = predictions_to_events(
        prediction=pred, meta_path=meta_path, behavior=args.behavior,
        threshold=args.threshold, min_duration_s=args.min_duration_s,
        min_gap_s=args.min_gap_s, fps=args.fps)
    out = Path(args.out) if args.out else (
        paths.events / f"{Path(pred).name}_events.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Wrote {len(df)} event(s) -> {out}")
    if len(df):
        print(f"Events per video: {df.groupby('video').size().to_dict()}")


def cmd_score(args):
    paths = Paths(args.workdir)
    pred = paths.prediction_dir(args.project_name, args.prediction)
    split = paths.split_file(args.split)
    r = score_predictions(
        prediction=pred, data_dir=paths.data, split_path=split,
        behavior=args.behavior, threshold=args.threshold,
        min_duration_s=args.min_duration_s, min_gap_s=args.min_gap_s,
        iou=args.iou, fps=args.fps)
    print(f"\nthreshold {r['threshold']} (chosen on: {r['chosen_on']})")
    keys = [("frame_precision", "frame P"), ("frame_recall", "frame R"),
            ("frame_f1", "frame F1"), ("event_recall", "bout recall"),
            ("event_precision", "event P"), ("bouts_detected", "bouts hit"),
            ("bouts_missed", "missed"), ("false_events", "false events"),
            ("false_events_per_hour", "FP/hour bg")]
    print(f"{'':6s}" + "".join(f"{lab:>14s}" for _, lab in keys))
    for sub in ("val", "test"):
        m = r[sub]
        if m is None:
            print(f"{sub:6s}  (nothing scored)")
            continue
        print(f"{sub:6s}" + "".join(
            f"{m[k]:14.3f}" if isinstance(m[k], float) else f"{m[k]:14d}"
            for k, _ in keys))
    out = paths.events / f"{Path(pred).name}__{split.stem}_score.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=2, default=float), encoding="utf-8")
    print(f"\nreport -> {out}")




def add_training_args(p):
    p.add_argument("--project-name",
                   default=env_default("DLC2ACTION_PROJECT", "cichlid_circling"))
    p.add_argument("--split", default=None,
                   help="split name in dlc2action/splits (e.g. fold0) or path; "
                        "required with --partition-method file (default)")
    p.add_argument("--partition-method", default="file",
                   help="file (default, use --split) | time:strict | time | "
                        "val-from-name:<v>:test-from-name:<t>")
    p.add_argument("--episode", default=None,
                   help="episode name (default: the split name)")
    p.add_argument("--behaviors", default="circling")
    p.add_argument("--canvas", default=env_default("DLC2ACTION_CANVAS", "1280x720"),
                   help="video WIDTHxHEIGHT in pixels, e.g. 1280x720")
    p.add_argument("--len-segment", type=int, default=128)
    p.add_argument("--overlap", type=int, default=64)
    p.add_argument("--min-frames-action", type=int, default=5,
                   help="shorter annotated bouts are ignored in training")
    p.add_argument("--val-frac", type=float, default=0.15,
                   help="only for time/time:strict partitions")
    p.add_argument("--test-frac", type=float, default=0.15,
                   help="only for time/time:strict partitions")
    p.add_argument("--num-epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default=env_default("DLC2ACTION_DEVICE", "auto"))
    p.add_argument("--num-cpus", type=int,
                   default=int(env_default("DLC2ACTION_NUM_CPUS", "8")))
    p.add_argument("--likelihood-threshold", type=float, default=0.5,
                   help="must be > 0: missing keypoints have likelihood 0")
    p.add_argument("--overwrite", action="store_true",
                   help="replace an existing episode/search with this name")
    p.add_argument("--pair-features", action="store_true",
                   help="also feed the pair-parquet features (run "
                        "convert-pairs first); episode becomes <split>_pair")
    p.add_argument("--feature-file", choices=list(FEATURE_FILES), default=None,
                   help="feed a combined feature file instead of the pair "
                        "file: pairemb = pair features + fish-mask embedding "
                        "PCs (embed_probe.py features). Exclusive with "
                        "--pair-features; episode becomes <split>_<name>")
    p.add_argument("--preset", action="append", choices=sorted(PRESETS),
                   default=None,
                   help="experiment preset, repeatable: aug (rotate 360, "
                        "mirror x+y, shift, noise), ssl (prediction generator "
                        "as feature extractor + masked_frames + contrastive), "
                        "nocenter (drop absolute position). Adds _<preset> to "
                        "the episode name")
    p.add_argument("--param", action="append", default=None,
                   metavar="SECTION.KEY=VALUE",
                   help="any DLC2Action parameter, YAML value, repeatable: "
                        "model.num_f_maps=64, training.lr=5e-4, "
                        "general.ssl=[masked_frames]. Needs --tag")
    p.add_argument("--tag", default=None,
                   help="extra episode-name suffix (required with --param)")


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"),
                    choices=["DEBUG", "INFO", "WARNING"])
    ap.add_argument("--workdir", default=None,
                    help="override <repo>/outputs/mstcn_dlc2action "
                         "(or set DLC2ACTION_WORKDIR)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("convert-data", help="single-fish parquet -> H5")
    p.add_argument("--keypoints", nargs="+", required=True,
                   help="single-fish (per-track) keypoint parquet(s)")
    p.add_argument("--ground-truth", type=Path,
                   default=env_default("DLC2ACTION_GT", None),
                   help="used to match parquet names to video ids")
    p.add_argument("--name-map", nargs="*", default=None, metavar="OLD=NEW",
                   help="explicit renames, e.g. 0028_keypoints=0028_vid")
    p.add_argument("--scorer", default=os.getenv("SCORER", "YOLO"))
    p.add_argument("--slot-assignment", default="male-partner",
                   choices=["male-partner", "tracklet", "rank"],
                   help="male-partner (default): ind0 = the male, ind1 = the "
                        "nearest female; tracklet: tank with exactly 2 fish")
    p.add_argument("--male-values", nargs="+", default=["m", "male"],
                   help="Sex column values meaning male (case-insensitive)")
    p.add_argument("--switch-factor", type=float, default=1.5,
                   help="switch partner only when another female is this "
                        "many times closer than the current one")
    p.add_argument("--max-gap", type=int, default=15)
    p.add_argument("--max-individuals", type=int, default=2,
                   help="slots per video (keep 2: male + partner)")
    p.add_argument("--frame-offset", type=int,
                   default=int(os.getenv("FRAME_OFFSET", "0")))
    p.add_argument("--stride", type=int,
                   default=int(env_default("DLC2ACTION_STRIDE", "1")),
                   help="keep every Nth frame (2-3 cuts training time "
                        "proportionally; annotations adjust automatically)")
    p.add_argument("--no-float32", action="store_true")
    p.add_argument("--no-compress", action="store_true")
    p.set_defaults(fn=cmd_convert_data)

    p = sub.add_parser("convert-ann", help="ground truth -> csv annotations")
    p.add_argument("--ground-truth", type=Path,
                   default=env_default("DLC2ACTION_GT", None),
                   required=not os.getenv("DLC2ACTION_GT"),
                   help="CSV: Video,StartFrame,StopFrame[,label]")
    p.add_argument("--label", default="circling")
    p.add_argument("--fps", type=float, default=None,
                   help="only for time-based ground truth")
    p.add_argument("--merge", action="store_true",
                   help="merge with existing csvs (e.g. extra negatives)")
    p.set_defaults(fn=cmd_convert_ann)

    p = sub.add_parser("convert-pairs",
                       help="pair parquet -> pair features for DLC2Action")
    p.add_argument("--pairs", nargs="+", required=True,
                   help="pair parquet(s), one per video")
    p.add_argument("--name-map", nargs="*", default=None, metavar="OLD=NEW",
                   help="explicit renames if names don't auto-match")
    p.set_defaults(fn=cmd_convert_pairs)

    p = sub.add_parser("make-split", help="event-aware train/val/test split")
    p.add_argument("--scheme", choices=["cluster", "video", "crossvideo"],
                   default="cluster",
                   help="cluster: leave one bout cluster out; video: whole "
                        "videos (>= 3); crossvideo: fold k tests on the whole "
                        "video k, trains on the others (team protocol)")
    p.add_argument("--fold", default="0",
                   help="cluster index (cluster) or video index (crossvideo) "
                        "for the test set, or 'all'")
    p.add_argument("--val-frac-split", type=float, default=0.25,
                   help="crossvideo: share of training positives held out as "
                        "the validation tail")
    p.add_argument("--val-cluster", type=int, default=None)
    p.add_argument("--name", default=None,
                   help="split name; may contain {k} for the fold, e.g. "
                        "fold{k}_l256 (keeps 'foldK' at the start so the "
                        "report tools find the fold)")
    p.add_argument("--list-clusters", action="store_true")
    p.add_argument("--cluster-gap", type=int, default=5000,
                   help="bouts closer than this many H5 rows form a cluster")
    p.add_argument("--context", type=int, default=9000,
                   help="background rows on each side of a held-out cluster")
    p.add_argument("--guard", type=int, default=None,
                   help="excluded rows around held-out windows "
                        "(default len_segment)")
    p.add_argument("--test-videos", nargs="*")
    p.add_argument("--val-videos", nargs="*")
    p.add_argument("--train-background-frac", type=float, default=1.0,
                   help="<1 keeps only this fraction of far-from-bout train "
                        "segments (much faster training)")
    p.add_argument("--near-bout", type=int, default=3000)
    p.add_argument("--len-segment", type=int, default=128)
    p.add_argument("--overlap", type=int, default=64)
    p.add_argument("--behavior", default="circling")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_make_split)

    p = sub.add_parser("train", help="create/update project and train")
    add_training_args(p)
    p.add_argument("--search", action="store_true",
                   help="hyperparameter search first (very slow)")
    p.add_argument("--search-name", default="search")
    p.add_argument("--search-trials", type=int, default=30,
                   help="optuna trials (each is a full training run, pruned "
                        "early when hopeless)")
    p.add_argument("--skip-preflight", action="store_true")
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("predict", help="predict on a folder of H5 files")
    p.add_argument("--project-name",
                   default=env_default("DLC2ACTION_PROJECT", "cichlid_circling"))
    p.add_argument("--episode", required=True)
    p.add_argument("--prediction-name", default=None,
                   help="default <episode>_pred")
    p.add_argument("--data", default=None,
                   help="folder of H5 files (default: dlc2action/data)")
    p.add_argument("--augment-n", type=int, default=10,
                   help="test-time augmentations averaged per segment "
                        "(DLC2Action default 10; 0 = 10x faster)")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(fn=cmd_predict)

    p = sub.add_parser("export-events", help="prediction -> events parquet")
    p.add_argument("--project-name",
                   default=env_default("DLC2ACTION_PROJECT", "cichlid_circling"))
    p.add_argument("--prediction", required=True,
                   help="prediction name or folder")
    p.add_argument("--out", default=None)
    p.add_argument("--meta", default=None)
    p.add_argument("--behavior", default="circling")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--min-duration-s", type=float, default=1.0)
    p.add_argument("--min-gap-s", type=float, default=0.0)
    p.add_argument("--fps", type=float, default=None,
                   help="effective (post-stride) fps; overrides metadata")
    p.set_defaults(fn=cmd_export_events)

    p = sub.add_parser("score", help="metrics on held-out val/test windows")
    p.add_argument("--project-name",
                   default=env_default("DLC2ACTION_PROJECT", "cichlid_circling"))
    p.add_argument("--prediction", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--behavior", default="circling")
    p.add_argument("--threshold", type=float, default=None,
                   help="fixed threshold (default: tuned on val)")
    p.add_argument("--min-duration-s", type=float, default=1.0)
    p.add_argument("--min-gap-s", type=float, default=0.0)
    p.add_argument("--iou", type=float, default=0.3)
    p.add_argument("--fps", type=float, default=None)
    p.set_defaults(fn=cmd_score)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(message)s")
    warn_legacy_env()
    if getattr(args, "partition_method", None) == "file" and \
            args.cmd == "train" and not args.split:
        raise SystemExit("train: --split is required with --partition-method "
                         "file (run make-split first)")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
DINOv2 embeddings for the MS-TCN circling model (models/mstcn_dlc2action).

  extract-full  EVERY H5 row (or every --row-step-th) of each video: crop the
            video around the male (ind0), centred on his centroid, rotated so
            his head points up, side = --crop-bl x his median body length,
            fish mask by default; embed with a frozen DINOv2 (CLS token).
            Streams the video once (decode in a background thread), writes
            <out>/<video>_embfull.npy (float16, NaN where the male pose is
            unusable) and resumes after an interruption.

  features  Builds the DLC2Action feature file <data>/<video>_pairemb.pickle:
            the 17 pair features + --pca embedding PCs (per-video centred
            unless --no-center, PCA fit label-free on a sample of rows of
            all videos, whitened, clipped at +-6) + emb_present.

  probe     Per-frame linear probe on the same data and split as the MS-TCN:
            the repository's logistic regression pipeline
            (models/logistic_regression/train.new_model) trained on every
            row of the TRAIN segments of repo_split, scored with the
            repository's metrics() on every frame of the TEST segments, i.e.
            the same test frames and labels as outputs/mstcn_dinov2/metrics.json.
            Column sets: pair (first --pair-cols), emb (the rest), all.
            Answers: do the embeddings add anything over the pair features,
            and how much does the MS-TCN's temporal context add?

`python -m models.mstcn_dlc2action.train` runs extract-full and features
itself when the feature files are missing. Standalone, from the repository root:

  python models/mstcn_dlc2action/scripts/embed_probe.py extract-full \\
      --video 0028_vid=data/0028_vid.mp4 --video 0031_vid=data/0031_vid.mp4
  python models/mstcn_dlc2action/scripts/embed_probe.py features
  python models/mstcn_dlc2action/scripts/embed_probe.py probe        # after train

Folders default to outputs/mstcn_dlc2action/ (data/, emb_fish_full/).
Needs torch + the DINOv2 weights (torch.hub downloads them once, ~90 MB for
ViT-S/14), opencv-python, scikit-learn, pandas, tables; probe also polars.
"""

import argparse
import json
import pickle
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PAIR_CLIP = "ind0+ind1"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
REPO = Path(__file__).resolve().parents[3]
WORKDIR = REPO / "outputs" / "mstcn_dlc2action"
OUTPUT_DIR = REPO / "outputs" / "mstcn_dinov2"


# --------------------------------------------------------------------------- shared

def load_meta(data_dir):
    return json.loads(Path(data_dir, "_conversion_meta.json").read_text(encoding="utf-8"))



def row_labels(data_dir, video, n):
    """Per H5 row: 1 circling, 0 other, -1 unknown (from data/<video>.csv)."""
    lab = np.zeros(n, np.int8)
    p = Path(data_dir, f"{video}.csv")
    if not p.exists():
        sys.exit(f"missing {p}: run python -m models.mstcn_dlc2action.train (it writes the labels)")
    d = pd.read_csv(p)
    for b, s, e in zip(d["behavior"], d["start"], d["end"]):
        if b == "circling":
            lab[int(s):int(e)] = 1
    for b, s, e in zip(d["behavior"], d["start"], d["end"]):
        if b == "unknown":
            lab[int(s):int(e)] = -1
    return lab


# --------------------------------------------------------------------------- extract

def load_male_pose(h5, lik=0.5, ind="ind0", df=None):
    """(xy[rows, bodyparts, 2] with NaN for missing, bodypart names) for one slot."""
    if df is None:
        df = pd.read_hdf(h5)
    if df.columns.nlevels == 4:
        df.columns = df.columns.droplevel(0)
    bps = list(dict.fromkeys(df.columns.get_level_values(1)))
    xy = np.full((len(df), len(bps), 2), np.nan, np.float32)
    for b, bp in enumerate(bps):
        x = df[(ind, bp, "x")].to_numpy(np.float32)
        y = df[(ind, bp, "y")].to_numpy(np.float32)
        ok = df[(ind, bp, "likelihood")].to_numpy(np.float32) >= lik
        xy[ok, b, 0], xy[ok, b, 1] = x[ok], y[ok]
    return xy, bps


def male_geometry(xy, bps):
    """Centroid, unit heading (tail -> head) and validity per row."""
    idx = {b: i for i, b in enumerate(bps)}
    valid = np.isfinite(xy[..., 0])
    cnt = valid.sum(1, keepdims=True)
    cen = np.where(cnt > 0, np.nansum(xy, axis=1) / np.maximum(cnt, 1), np.nan)
    head = np.full((len(xy), 2), np.nan, np.float32)
    # most reliable axis first: head/peduncle, then nose/tailtip, then spine
    for front, back in (("Head", "Peduncle"), ("Nose", "TailTip"),
                        ("Spine1", "Spine4"), ("Nose", "Peduncle"),
                        ("Head", "TailTip")):
        if front in idx and back in idx:
            v = xy[:, idx[front]] - xy[:, idx[back]]
            fill = np.isnan(head[:, 0]) & np.isfinite(v).all(1)
            head[fill] = v[fill]
    norm = np.linalg.norm(head, axis=1, keepdims=True)
    with np.errstate(all="ignore"):
        head = head / norm
    ok = np.isfinite(cen).all(1) & np.isfinite(head).all(1) & (norm[:, 0] > 1)
    return cen, head, ok


def body_length(xy, bps):
    idx = {b: i for i, b in enumerate(bps)}
    for a, b in (("Nose", "TailTip"), ("Head", "Peduncle")):
        if a in idx and b in idx:
            d = np.linalg.norm(xy[:, idx[a]] - xy[:, idx[b]], axis=1)
            d = d[np.isfinite(d) & (d > 1)]
            if len(d) > 100:
                return float(np.median(d)), f"{a}-{b}"
    sys.exit("cannot estimate the male's body length from the H5")


def crop(frame, c, h, side, size):
    """Square crop centred at c, rotated so heading h points up."""
    k = side / size
    up = h
    down = -up
    right = np.array([-up[1], up[0]], np.float32)   # image coords, y down
    m = np.array([
        [k * right[0], k * down[0], c[0] - size / 2 * k * (right[0] + down[0])],
        [k * right[1], k * down[1], c[1] - size / 2 * k * (right[1] + down[1])],
    ], np.float32)
    out = cv2.warpAffine(frame, m, (size, size),
                         flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=MASK_FILL)
    return out, m


MASK_FILL = (128, 128, 128)          # neutral grey for everything masked
BODY_CHAIN = ["Nose", "Head", "Spine1", "Spine2", "Spine3", "Spine4",
              "Peduncle", "TailTip"]


def apply_mask(img, mode, m, fish_xy, bps, bl, side, a):
    if mode == "none":
        return img
    size = img.shape[0]
    px = size / side                             
    mask = np.zeros(img.shape[:2], np.uint8)
    if mode == "circle":
        cv2.circle(mask, (size // 2, size // 2), int(a.mask_radius_bl * bl * px), 255, -1)
    else:
        inv = cv2.invertAffineTransform(m)        # video pixels -> crop pixels
        idx = {b: i for i, b in enumerate(bps)}
        chain = [idx[b] for b in BODY_CHAIN if b in idx] or list(range(len(bps)))
        width = max(2, int(a.fish_width_bl * bl * px))
        for pts in fish_xy:                      
            p = pts[chain]
            p = p[np.isfinite(p).all(1)]
            if len(p) == 0:
                continue
            q = (np.c_[p, np.ones(len(p))] @ inv.T).astype(np.int32)
            if len(q) > 1:
                cv2.polylines(mask, [q.reshape(-1, 1, 2)], False, 255, width, cv2.LINE_AA)
            for x, y in q:
                cv2.circle(mask, (int(x), int(y)), width // 2, 255, -1)
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    out = img.copy()
    out[mask == 0] = MASK_FILL
    return out


class FrameReader:
    def __init__(self, path, max_skip=600):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            sys.exit(f"cannot open video {path}")
        self.pos = 0
        self.max_skip = max_skip
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def read(self, f):
        if not 0 <= f - self.pos <= self.max_skip:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            self.pos = f
        while self.pos < f:
            self.cap.grab()
            self.pos += 1
        ok, img = self.cap.read()
        self.pos += 1
        return img if ok else None


def load_encoder(name, device):
    import torch
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", name)
    model.eval().to(device)
    half = device.startswith("cuda")
    if half:
        model.half()

    def encode(batch_rgb):
        x = np.stack(batch_rgb).astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        t = torch.from_numpy(x).permute(0, 3, 1, 2).to(device)
        if half:
            t = t.half()
        with torch.no_grad():
            out = model(t)
        return out.float().cpu().numpy()

    print(f"encoder {name} on {device}{' (fp16)' if half else ''}")
    return encode




def cmd_extract_full(a):
    """Embed every --row-step-th H5 row of each video (resumable)."""
    meta = load_meta(a.data_dir)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    videos = dict((s.strip() for s in e.split("=", 1)) for e in a.video)
    encode = None
    for v, vpath in videos.items():
        if v not in meta:
            sys.exit(f"{v}: not in {a.data_dir}\\_conversion_meta.json")
        m = meta[v]
        n, stride, fmin = int(m["n_frames"]), int(m.get("stride", 1)), int(m.get("frame_min", 0))
        npy, info_p = out / f"{v}_embfull.npy", out / f"{v}_embfull.json"
        settings = {"video": v, "encoder": a.encoder, "mask": a.mask, "crop_bl": a.crop_bl,
                    "size": a.size, "fish_width_bl": a.fish_width_bl,
                    "mask_radius_bl": a.mask_radius_bl, "row_step": a.row_step,
                    "n_rows": n, "stride": stride, "frame_min": fmin}
        info = json.loads(info_p.read_text(encoding="utf-8")) if info_p.exists() else None
        if info is not None and not a.overwrite:
            if any(info.get(k) != val for k, val in settings.items()):
                sys.exit(f"[{v}] {info_p.name} was made with other settings; "
                         "use another --out or --overwrite")
            if info.get("complete"):
                print(f"[{v}] complete, skipping (--overwrite to redo)")
                continue
        if info is None or a.overwrite:
            info = dict(settings, next_row=0, complete=False)
        h5df = pd.read_hdf(Path(a.data_dir, f"{v}.h5"))
        if len(h5df) != n:
            sys.exit(f"[{v}] H5 has {len(h5df)} rows, meta says {n}: re-convert")
        xy, bps = load_male_pose(None, df=h5df)
        xy_f = load_male_pose(None, ind="ind1", df=h5df)[0] if a.mask == "fish" else None
        del h5df
        cen, head, ok = male_geometry(xy, bps)
        bl, bl_def = body_length(xy, bps)
        side = a.crop_bl * bl
        reader = FrameReader(vpath)
        W, H = reader.size
        if np.nanmax(xy[..., 0]) > W * 1.02 or np.nanmax(xy[..., 1]) > H * 1.02:
            sys.exit(f"[{v}] keypoints exceed the {W}x{H} video: H5 coordinates are "
                     "not in video pixels")
        if (n - 1) * stride + fmin >= reader.n > 0:
            sys.exit(f"[{v}] H5 rows map to frame {(n - 1) * stride + fmin} but the video "
                     f"has {reader.n} frames: wrong video file or stride")
        if encode is None:
            encode = load_encoder(a.encoder, a.device)
        dim = encode([np.zeros((a.size, a.size, 3), np.uint8)]).shape[1]
        if info["next_row"] == 0 or not npy.exists():
            emb = np.lib.format.open_memmap(npy, mode="w+", dtype=np.float16, shape=(n, dim))
            emb[:] = np.nan
            emb.flush()
            info["next_row"] = 0
        else:
            emb = np.load(npy, mmap_mode="r+")
            if emb.shape != (n, dim):
                sys.exit(f"[{v}] {npy.name} has shape {emb.shape}, expected {(n, dim)}")
        info.update(body_length_px=bl, body_length_def=bl_def, crop_side_px=side, dim=int(dim))
        start = info["next_row"]
        rows = np.arange(start - start % a.row_step, n, a.row_step)
        rows = rows[ok[rows]]
        print(f"[{v}] mask {a.mask}; {len(rows)} rows to embed from row {start} "
              f"({100 * ok[::a.row_step].mean():.1f}% of rows have a usable male pose); "
              f"body length {bl:.0f} px, crop side {side:.0f} px")

        q = queue.Queue(maxsize=8)
        err = []

        def produce():
            try:
                batch, brows = [], []
                for r in rows:
                    img = reader.read(int(r) * stride + fmin)
                    if img is None:
                        continue
                    c, aff = crop(img, cen[r], head[r], side, a.size)
                    c = apply_mask(c, a.mask, aff,
                                   [xy[r]] + ([xy_f[r]] if xy_f is not None else []),
                                   bps, bl, side, a)
                    batch.append(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
                    brows.append(int(r))
                    if len(batch) == a.batch:
                        q.put((brows, batch))
                        batch, brows = [], []
                if batch:
                    q.put((brows, batch))
            except Exception as ex:     
                err.append(ex)
            finally:
                q.put(None)

        th = threading.Thread(target=produce, daemon=True)
        th.start()
        t0, done, since_flush = time.time(), 0, 0
        while True:
            item = q.get()
            if item is None:
                break
            brows, batch = item
            emb[brows] = encode(batch).astype(np.float16)
            done += len(brows)
            since_flush += len(brows)
            if since_flush >= a.chunk:
                emb.flush()
                info["next_row"] = brows[-1] + 1
                info_p.write_text(json.dumps(info, indent=2), encoding="utf-8")
                since_flush = 0
                rate = done / (time.time() - t0)
                print(f"  row {brows[-1] + 1}/{n}  {rate:.0f} crops/s, "
                      f"~{(len(rows) - done) / max(rate, 1e-9) / 60:.0f} min left", flush=True)
        th.join()
        if err:
            emb.flush()
            info_p.write_text(json.dumps(info, indent=2), encoding="utf-8")
            raise err[0]
        emb.flush()
        info.update(next_row=n, complete=True,
                    frac_rows_embedded=round(float(np.isfinite(emb[:, 0]).mean()), 4))
        info_p.write_text(json.dumps(info, indent=2), encoding="utf-8")
        del emb
        print(f"[{v}] done: {done} crops in {(time.time() - t0) / 60:.0f} min -> {npy}")

def cmd_features(a):
    from sklearn.decomposition import PCA
    meta = load_meta(a.data_dir)
    rng = np.random.default_rng(a.seed)
    missing = [v for v in meta if not Path(a.emb_dir, f"{v}_embfull.json").exists()]
    if missing:
        sys.exit(f"no full embeddings for {missing} in {a.emb_dir}: DLC2Action would skip "
                 "those videos. Run extract-full for every video in the data folder.")
    embs, infos = {}, {}
    for v in sorted(meta):
        info = json.loads(Path(a.emb_dir, f"{v}_embfull.json").read_text(encoding="utf-8"))
        if not info.get("complete"):
            sys.exit(f"[{v}] extract-full did not finish (next_row {info.get('next_row')}); "
                     "re-run it to resume")
        e = np.load(Path(a.emb_dir, f"{v}_embfull.npy"), mmap_mode="r")
        if e.shape[0] != int(meta[v]["n_frames"]):
            sys.exit(f"[{v}] embeddings have {e.shape[0]} rows, H5 has {meta[v]['n_frames']}")
        embs[v], infos[v] = e, info
    masks = {i["mask"] for i in infos.values()}
    if len(masks) != 1:
        sys.exit(f"videos were embedded with different masks: {masks}")
    CH = 50_000

    def valid_rows(v):
        e, step = embs[v], int(infos[v].get("row_step", 1))
        ok = np.zeros(len(e), bool)
        for i in range(0, len(e), CH):
            ok[i:i + CH] = np.isfinite(np.asarray(e[i:i + CH, 0], np.float32))
        src = np.arange(len(e)) - np.arange(len(e)) % step   
        return ok, src

    means, sample = {}, []
    per_video = a.sample // len(embs)
    vrows = {}
    for v, e in embs.items():
        ok, src = valid_rows(v)
        vrows[v] = (ok, src)
        idx = np.flatnonzero(ok)
        if len(idx) == 0:
            sys.exit(f"[{v}] no embedded rows")
        acc = np.zeros(e.shape[1], np.float64)
        for i in range(0, len(idx), CH):
            acc += np.asarray(e[idx[i:i + CH]], np.float64).sum(0)
        means[v] = (acc / len(idx)).astype(np.float32) if a.center else np.zeros(e.shape[1], np.float32)
        pick = np.sort(rng.choice(idx, min(per_video, len(idx)), replace=False))
        sample.append(np.asarray(e[pick], np.float32) - means[v])
    sample = np.concatenate(sample)
    k = min(a.pca, sample.shape[1])
    pca = PCA(n_components=k, svd_solver="randomized", random_state=a.seed).fit(sample)
    scale = np.sqrt(pca.explained_variance_).astype(np.float32)
    print(f"PCA {k} comps on {len(sample)} rows ({'per-video centred' if a.center else 'raw'}): "
          f"{100 * pca.explained_variance_ratio_.sum():.1f}% of variance")

    names = None
    summary = {}
    for v, e in embs.items():
        pf = Path(a.data_dir, f"{v}_pairfeat.pickle")
        pnames = (meta[v].get("pair_stats") or {}).get("pair_features")
        if not pf.exists() or not pnames:
            sys.exit(f"{v}: no pair features ({pf}); run python -m models.mstcn_dlc2action.train "
                     "(it builds them) or convert-pairs")
        with open(pf, "rb") as fh:
            pair = np.asarray(pickle.load(fh)[PAIR_CLIP], np.float32)
        n = len(e)
        if len(pair) != n:
            sys.exit(f"[{v}] pair file has {len(pair)} rows, embeddings {n}")
        ok, src = vrows[v]
        present = ok[src]
        pcs = np.zeros((n, k), np.float32)
        for i in range(0, n, CH):
            s_ = src[i:i + CH]
            pr = present[i:i + CH]
            if pr.any():
                x = np.asarray(e[s_[pr]], np.float32) - means[v]
                pcs[i:i + CH][pr] = np.clip(pca.transform(x) / scale, -6, 6)
        out = np.column_stack([pair, pcs, present.astype(np.float32)]).astype(np.float32)
        names = list(pnames) + [f"emb_pc{j + 1}" for j in range(k)] + ["emb_present"]
        dst = Path(a.data_dir, f"{v}_{a.name}.pickle")
        with open(dst, "wb") as fh:
            pickle.dump({PAIR_CLIP: out}, fh)
        lab_p = Path(a.data_dir, f"{v}.csv")
        cov_bout = None
        if lab_p.exists():
            lab = row_labels(a.data_dir, v, n)
            if (lab == 1).any():
                cov_bout = round(float(present[lab == 1].mean()), 4)
        summary[v] = {"rows": n, "emb_present": round(float(present.mean()), 4),
                      "emb_present_in_bouts": cov_bout,
                      "pair_present": round(float(pair[:, -1].mean()), 4)}
        print(f"[{v}] {out.shape} -> {dst.name}; embedding present in "
              f"{100 * present.mean():.1f}% of rows"
              + (f", {100 * cov_bout:.1f}% of circling rows" if cov_bout is not None else ""))
    np.savez(Path(a.data_dir, f"_{a.name}_pca.npz"), components=pca.components_,
             pca_mean=pca.mean_, scale=scale,
             **{f"video_mean__{v}": m for v, m in means.items()})
    Path(a.data_dir, f"_{a.name}_info.json").write_text(json.dumps({
        "feature_names": names, "pca": k, "center_per_video": bool(a.center),
        "pca_sample_rows": int(len(sample)), "seed": a.seed,
        "explained_variance": round(float(pca.explained_variance_ratio_.sum()), 4),
        "emb_dir": str(Path(a.emb_dir).resolve()), "extraction": next(iter(infos.values())),
        "videos": summary,
        "note": ("PCA and per-video means are label-free and fit on all rows of these "
                 "videos. A NEW video: extract-full it, centre on its own mean, project "
                 "with the saved components (re-running features refits the PCA, which "
                 "changes the inputs of trained models)."),
    }, indent=2), encoding="utf-8")
    print(f"\nNext: python -m models.mstcn_dlc2action.train (uses <video>_{a.name}.pickle). "
          "Do NOT re-run features after training: it refits the PCA.")




def cmd_probe(a):
    sys.path.insert(0, str(REPO))
    import polars as pl
    from models.logistic_regression.train import VIDEO_FRAMES, fit_model, metrics
    from models.mstcn_dlc2action import train as mstcn

    meta = load_meta(a.data_dir)
    seg_path = Path(a.output_dir, "sampled_segments.parquet")
    spans_csv = Path(a.workdir, "training_spans.csv")
    for p in (seg_path, spans_csv):
        if not p.exists():
            sys.exit(f"missing {p}: run python -m models.mstcn_dlc2action.train first")
    segments = pl.read_parquet(seg_path)
    spans = mstcn.read_annotations(spans_csv)        
    X, Y = {}, {}
    for v in VIDEO_FRAMES:
        p = Path(a.data_dir, f"{v}_{a.feature_file}.pickle")
        if not p.exists():
            sys.exit(f"missing {p}")
        with open(p, "rb") as fh:
            X[v] = np.asarray(pickle.load(fh)[PAIR_CLIP], np.float32)
        Y[v] = row_labels(a.data_dir, v, len(X[v]))
    d = next(iter(X.values())).shape[1]
    sets = {"pair": slice(0, a.pair_cols), "emb": slice(a.pair_cols, d), "all": slice(0, d)}

    def rows(split):
        return [(v, np.arange(s, e)) for v, s, e in segments.filter(pl.col("split") == split)
                .select("video", "start_row", "end_row").iter_rows()]

    train_rows = [(v, r[Y[v][r] >= 0]) for v, r in rows("train")]
    ytr = np.concatenate([Y[v][r] for v, r in train_rows]).astype(int)
    results = []
    for name, sl in sets.items():
        if sl.start >= sl.stop:
            continue
        model = fit_model(np.concatenate([X[v][r, sl] for v, r in train_rows]), ytr)
        probs = {v: np.zeros(len(X[v])) for v in X}
        for v, r in rows("test"):
            probs[v][r] = model.predict_proba(X[v][r, sl])[:, 1]
        scores = {v: mstcn.frame_scores(probs[v], meta[v], VIDEO_FRAMES[v]) for v in X}
        test = mstcn.test_frames(segments, meta, scores, spans)
        result = metrics(test["circling"].to_numpy(), test["circling_score"].to_numpy())
        results.append({"features": name, "columns": sl.stop - sl.start, **result})
        print(f"{name}: AP {result['average_precision']:.3f}, ROC AUC {result['roc_auc']:.3f}, "
              f"F1@0.5 {result['f1']:.3f}", flush=True)
    res = pd.DataFrame(results)
    mstcn_metrics = Path(a.output_dir, "metrics.json")
    if mstcn_metrics.exists():
        t = json.loads(mstcn_metrics.read_text(encoding="utf-8"))["test"]
        res = pd.concat([res, pd.DataFrame([{"features": "MS-TCN (metrics.json)", **t}])])
    print("\nTest frames of repo_split (chance AP = prevalence):")
    print(res[["features", "prevalence", "average_precision", "roc_auc", "precision", "recall", "f1"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    out = Path(a.output_dir, f"probe_{a.feature_file}.csv")
    res.to_csv(out, index=False)
    print(f"\nSaved {out}")



def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("extract-full", help="crop + embed EVERY row (for training)")
    f.add_argument("--data-dir", default=str(WORKDIR / "data"))
    f.add_argument("--video", action="append", required=True, metavar="NAME=PATH")
    f.add_argument("--out", default=str(WORKDIR / "emb_fish_full"))
    f.add_argument("--encoder", default="dinov2_vits14",
                   help="torch.hub DINOv2 model (dinov2_vits14 / vitb14 ...)")
    f.add_argument("--device", default="auto")
    f.add_argument("--mask", choices=["none", "circle", "fish"], default="fish",
                   help="grey out the background: circle = keep a disc around "
                        "the male; fish = keep only the male's and partner's "
                        "bodies (from keypoints)")
    f.add_argument("--mask-radius-bl", type=float, default=1.25,
                   help="circle mask radius in male body lengths")
    f.add_argument("--fish-width-bl", type=float, default=0.3,
                   help="fish mask body width in male body lengths")
    f.add_argument("--crop-bl", type=float, default=3.0,
                   help="crop side in male body lengths")
    f.add_argument("--size", type=int, default=224, help="crop pixels (multiple of 14)")
    f.add_argument("--row-step", type=int, default=1,
                   help="embed every Nth H5 row; rows in between reuse the last one")
    f.add_argument("--batch", type=int, default=128)
    f.add_argument("--chunk", type=int, default=20000,
                   help="save progress every N rows (resume point)")
    f.add_argument("--overwrite", action="store_true")
    g = sub.add_parser("features", help="pair + embedding PCs -> <video>_pairemb.pickle")
    g.add_argument("--data-dir", default=str(WORKDIR / "data"))
    g.add_argument("--emb-dir", default=str(WORKDIR / "emb_fish_full"), help="extract-full output")
    g.add_argument("--pca", type=int, default=32)
    g.add_argument("--no-center", dest="center", action="store_false",
                   help="skip per-video centring (default: centre each video on its own mean)")
    g.add_argument("--sample", type=int, default=200000, help="rows for the PCA fit")
    g.add_argument("--name", default="pairemb")
    g.add_argument("--seed", type=int, default=0)
    h = sub.add_parser("probe", aliases=["probe-file"],
                       help="per-frame logistic regression on the MS-TCN's split and test frames")
    h.add_argument("--workdir", default=str(WORKDIR))
    h.add_argument("--data-dir", default=None, help="default <workdir>/data")
    h.add_argument("--output-dir", default=str(OUTPUT_DIR),
                   help="train's output (sampled_segments.parquet, metrics.json)")
    h.add_argument("--feature-file", default="pairemb")
    h.add_argument("--pair-cols", type=int, default=17)
    a = ap.parse_args(argv)
    if getattr(a, "size", 14) % 14:
        sys.exit("--size must be a multiple of 14 for DINOv2")
    if a.cmd in ("probe", "probe-file") and a.data_dir is None:
        a.data_dir = str(Path(a.workdir, "data"))
    {"extract-full": cmd_extract_full, "features": cmd_features,
     "probe": cmd_probe, "probe-file": cmd_probe}[a.cmd](a)


if __name__ == "__main__":
    main()

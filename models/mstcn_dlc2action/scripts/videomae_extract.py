import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
WORKDIR = REPO / 'outputs' / 'mstcn_dlc2action'
MODEL = 'MCG-NJU/videomae-base-finetuned-kinetics'
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

def letterbox(frame, size):
    h, w = frame.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    out = np.zeros((size, size, 3), np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top:top + nh, left:left + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

def load_encoder(name, revision, device):
    import torch
    from transformers import VideoMAEModel
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VideoMAEModel.from_pretrained(name, revision=revision).eval().to(device)
    half = device.startswith('cuda')
    if half:
        model.half()

    def encode(clips):
        x = (np.stack(clips).astype(np.float32) / 255.0 - MEAN) / STD
        t = torch.from_numpy(x).permute(0, 1, 4, 2, 3).to(device)
        if half:
            t = t.half()
        with torch.no_grad():
            out = model(pixel_values=t).last_hidden_state.mean(1)
        return out.float().cpu().numpy()

    print(f'encoder {name}@{revision} on {device}{" (fp16)" if half else ""}', flush=True)
    return encode, model.config.hidden_size

def clip_frames(anchor, clip, step, last):
    return np.clip(anchor + step * (np.arange(clip) - clip // 2), 0, last)

def interpolate(anchors, n_rows, stride, first, anchor_step, out, chunk=50_000):
    n = len(anchors)
    for start in range(0, n_rows, chunk):
        rows = np.arange(start, min(n_rows, start + chunk))
        position = (rows * stride + first) / anchor_step
        lo = np.clip(np.floor(position).astype(np.int64), 0, n - 1)
        hi = np.clip(lo + 1, 0, n - 1)
        w = np.clip(position - lo, 0, 1)[:, None].astype(np.float32)
        out[rows] = ((1 - w) * anchors[lo].astype(np.float32) + w * anchors[hi].astype(np.float32)).astype(np.float16)

def extract(video, path, meta, a, encode, dim):
    out = Path(a.out)
    n_rows, stride, first = int(meta['n_frames']), int(meta.get('stride', 1)), int(meta.get('frame_min', 0))
    last = (n_rows - 1) * stride + first
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        sys.exit(f'cannot open video {path}')
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if 0 < total <= last:
        last = total - 1
    anchors_at = np.arange(0, last + 1, a.anchor_step)
    settings = {'video': video, 'encoder': f'{a.model}@{a.revision}', 'mask': 'none', 'size': a.size,
                'clip_frames': a.clip_frames, 'frame_step': a.frame_step, 'anchor_step': a.anchor_step,
                'row_step': 1, 'n_rows': n_rows, 'stride': stride, 'frame_min': first}
    info_path = out / f'{video}_embfull.json'
    info = json.loads(info_path.read_text(encoding='utf-8')) if info_path.exists() else None
    if info is not None and not a.overwrite:
        if any(info.get(k) != v for k, v in settings.items()):
            sys.exit(f'[{video}] {info_path.name} was made with other settings; use another --out or --overwrite')
        if info.get('complete'):
            print(f'[{video}] complete, skipping (--overwrite to redo)')
            capture.release()
            return
    if info is None or a.overwrite:
        info = dict(settings, next_anchor=0, complete=False)
    anchor_path = out / f'{video}_anchors.npy'
    if info['next_anchor'] == 0 or not anchor_path.exists():
        anchors = np.lib.format.open_memmap(anchor_path, mode='w+', dtype=np.float16,
                                            shape=(len(anchors_at), dim))
        info['next_anchor'] = 0
    else:
        anchors = np.load(anchor_path, mmap_mode='r+')
    start_anchor = info['next_anchor']
    todo = range(start_anchor, len(anchors_at))
    print(f'[{video}] {len(todo)} clips to embed from anchor {start_anchor} of {len(anchors_at)}', flush=True)

    clips = queue.Queue(maxsize=4)
    errors = []

    def produce():
        try:
            first_frame = int(clip_frames(anchors_at[start_anchor], a.clip_frames, a.frame_step, last)[0]) \
                if len(todo) else 0
            capture.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
            position, buffer, batch, ids = first_frame, {}, [], []
            for k in todo:
                needed = clip_frames(anchors_at[k], a.clip_frames, a.frame_step, last)
                wanted = set(int(f) for f in needed)
                while position <= needed.max():
                    if position in wanted:
                        ok, frame = capture.read()
                        if not ok:
                            raise RuntimeError(f'{video}: cannot decode frame {position}')
                        buffer[position] = letterbox(frame, a.size)
                    else:
                        if not capture.grab():
                            raise RuntimeError(f'{video}: cannot decode frame {position}')
                    position += 1
                batch.append(np.stack([buffer[int(f)] for f in needed]))
                ids.append(k)
                low = int(clip_frames(anchors_at[min(k + 1, len(anchors_at) - 1)], a.clip_frames,
                                      a.frame_step, last)[0])
                for f in [f for f in buffer if f < low]:
                    del buffer[f]
                if len(batch) == a.batch:
                    clips.put((ids, batch))
                    batch, ids = [], []
            if batch:
                clips.put((ids, batch))
        except Exception as error:
            errors.append(error)
        finally:
            clips.put(None)

    thread = threading.Thread(target=produce, daemon=True)
    thread.start()
    started, done, since_save = time.time(), 0, 0
    while True:
        item = clips.get()
        if item is None:
            break
        ids, batch = item
        anchors[ids] = encode(batch).astype(np.float16)
        done += len(ids)
        since_save += len(ids)
        if since_save >= a.save_every:
            anchors.flush()
            info['next_anchor'] = ids[-1] + 1
            info_path.write_text(json.dumps(info, indent=2), encoding='utf-8')
            since_save = 0
            rate = done / (time.time() - started)
            print(f'  clip {ids[-1] + 1}/{len(anchors_at)}  {rate:.1f} clips/s, '
                  f'~{(len(todo) - done) / max(rate, 1e-9) / 60:.0f} min left', flush=True)
    thread.join()
    capture.release()
    anchors.flush()
    if errors:
        info_path.write_text(json.dumps(info, indent=2), encoding='utf-8')
        raise errors[0]
    rows = np.lib.format.open_memmap(out / f'{video}_embfull.npy', mode='w+', dtype=np.float16, shape=(n_rows, dim))
    interpolate(np.load(anchor_path, mmap_mode='r'), n_rows, stride, first, a.anchor_step, rows)
    rows.flush()
    del rows
    info.update(next_anchor=len(anchors_at), complete=True, dim=int(dim), frac_rows_embedded=1.0)
    info_path.write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(f'[{video}] done: {done} clips in {(time.time() - started) / 60:.0f} min', flush=True)

def main(argv=None):
    parser = argparse.ArgumentParser(description='VideoMAE clip embeddings for every H5 row')
    parser.add_argument('--data-dir', default=str(WORKDIR / 'data'))
    parser.add_argument('--video', action='append', required=True, metavar='NAME=PATH')
    parser.add_argument('--out', default=str(WORKDIR / 'emb_videomae_full'))
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--revision', default='main', help='pin the Hugging Face commit for reproducible runs')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--clip-frames', type=int, default=16)
    parser.add_argument('--frame-step', type=int, default=2)
    parser.add_argument('--anchor-step', type=int, default=16)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--save-every', type=int, default=2000, help='clips between resume points')
    parser.add_argument('--overwrite', action='store_true')
    a = parser.parse_args(argv)
    meta = json.loads(Path(a.data_dir, '_conversion_meta.json').read_text(encoding='utf-8'))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    encode, dim = None, None
    for entry in a.video:
        video, path = (s.strip() for s in entry.split('=', 1))
        if video not in meta:
            sys.exit(f'{video}: not in {a.data_dir}/_conversion_meta.json')
        info = Path(a.out, f'{video}_embfull.json')
        if info.exists() and json.loads(info.read_text(encoding='utf-8')).get('complete') and not a.overwrite:
            print(f'[{video}] complete, skipping (--overwrite to redo)')
            continue
        if encode is None:
            encode, dim = load_encoder(a.model, a.revision, a.device)
        extract(video, path, meta[video], a, encode, dim)

if __name__ == '__main__':
    main()

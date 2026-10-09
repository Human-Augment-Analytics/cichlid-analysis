"""Fine-tune VideoMAE on seed clips, cache embeddings, and train DLC2Action MS-TCN.

0028_vid is the ONLY training video. 0031_vid is held out for both stages.
This is supervised adaptation of a pretrained encoder, not masked pretraining.
"""
import argparse
import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

from dlc2action_segmentation import ROOT, IGNORE, read_spans, source_hash, context_intervals, span_labels, run_test

OUTPUT = ROOT / 'outputs' / 'videomae_mstcn'
MODEL_ID = 'MCG-NJU/videomae-base-finetuned-kinetics'
REVISION = '488eb9a0565f257b32866000305c8178965eb9f6'
CLIP_FRAMES = 16
FRAME_STEP = 2
IMAGE_SIZE = 224
PIPELINE_VERSION = 1


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:20]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix('.tmp')
    partial.write_text(json.dumps(value, indent=2) + '\n')
    partial.replace(path)


def letterbox(frame):
    """Keep the entire scene and aspect ratio, padding to the encoder input size."""
    h, w = frame.shape[:2]
    ratio = IMAGE_SIZE / max(h, w)
    h2, w2 = round(h * ratio), round(w * ratio)
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    image[(IMAGE_SIZE-h2)//2:(IMAGE_SIZE-h2)//2+h2, (IMAGE_SIZE-w2)//2:(IMAGE_SIZE-w2)//2+w2] = cv2.cvtColor(
        cv2.resize(frame, (w2, h2), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return image


def clip_indices(center, count):
    """Sixteen frames at two-source-frame intervals, with edge replication."""
    return np.clip(center + np.arange(CLIP_FRAMES) - CLIP_FRAMES // 2, 0, count - 1)


def prepare_excerpts(data_dir, span_dir, output_dir, context):
    spans = read_spans(span_dir)
    if set(spans['Video ID']) != {'0028_vid', '0031_vid'}:
        raise ValueError('Expected the two seed videos 0028_vid and 0031_vid')
    videos = {v: Path(data_dir) / f'{v}.mp4' for v in sorted(set(spans['Video ID']))}
    source = {'version': PIPELINE_VERSION, 'span_sha256': source_hash(span_dir), 'context': context,
              'image_size': IMAGE_SIZE, 'frame_step': FRAME_STEP, 'resize': 'whole-frame RGB letterbox',
              'videos': {v: {'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for v, p in videos.items()}}
    cache = Path(output_dir) / 'frames' / fingerprint(source)
    cache.mkdir(parents=True, exist_ok=True)
    sequences = []
    for video, path in videos.items():
        rows = spans.filter(pl.col('Video ID') == video).to_dicts()
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ValueError(f'Cannot open {path}')
            last = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            intervals = context_intervals(rows, context, last)
            for index, (start, end) in enumerate(intervals):
                name = f'{video}-{index:03d}'
                frame_ids = np.arange(start, end + 1, FRAME_STEP)
                dest = cache / f'{name}.npy'
                signature = dict(video=video, sequence=name, start=start, end=end, fps=fps,
                                 cached_frames=len(frame_ids), source_key=fingerprint(source))
                meta = dest.with_suffix('.json')
                if not dest.exists() or not meta.exists() or json.loads(meta.read_text()) != signature:
                    partial = dest.with_suffix('.partial.npy')
                    images = np.lib.format.open_memmap(partial, mode='w+', dtype=np.uint8,
                                                       shape=(len(frame_ids), IMAGE_SIZE, IMAGE_SIZE, 3))
                    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
                    slot = 0
                    for frame_number in range(start, end + 1):
                        ok, frame = capture.read()
                        if not ok:
                            raise RuntimeError(f'Could not decode {video} frame {frame_number}')
                        if (frame_number - start) % FRAME_STEP == 0:
                            images[slot] = letterbox(frame)
                            slot += 1
                    images.flush()
                    del images
                    partial.replace(dest)
                    write_json(meta, signature)
                sequences.append({**signature, 'path': str(dest), 'frame_ids': frame_ids,
                                  'labels': span_labels(frame_ids, rows), 'rows': rows})
                print(f'Prepared {video} excerpt {index + 1}/{len(intervals)}: {start}–{end}', flush=True)
        finally:
            capture.release()
    write_json(cache / 'source.json', source)
    return sequences, source


class ClipDataset(Dataset):
    def __init__(self, sequences, examples, mean, std):
        self.arrays = [np.load(s['path'], mmap_mode='r') for s in sequences]
        self.examples = examples
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        sequence, center, label = self.examples[index]
        a = self.arrays[sequence]
        clip = np.array(a[clip_indices(center, len(a))], copy=True)
        pixels = torch.from_numpy(clip).permute(0, 3, 1, 2).float().div_(255)
        return (pixels - self.mean) / self.std, int(label)


def training_examples(sequences, samples, seed):
    """Class-balanced, sequence-round-robin sampling of homogeneous labeled clips."""
    rng = np.random.default_rng(seed)
    pools = {0: [], 1: []}
    for index, sequence in enumerate(sequences):
        if sequence['video'] != '0028_vid':
            continue
        labels = sequence['labels']
        centers = np.arange(CLIP_FRAMES // 2, len(labels) - CLIP_FRAMES // 2, 8)
        for label in (0, 1):
            candidates = [int(c) for c in centers if np.all(labels[clip_indices(c, len(labels))] == label)]
            rng.shuffle(candidates)
            if candidates:
                pools[label].append([(index, c, label) for c in candidates])
    result = []
    for label in (0, 1):
        groups = pools[label]
        rng.shuffle(groups)
        selected = []
        while len(selected) < samples // 2 and any(groups):
            for group in groups:
                if group and len(selected) < samples // 2:
                    selected.append(group.pop())
        if not selected:
            raise ValueError(f'No homogeneous training clips for class {label}')
        result.extend(selected)
    rng.shuffle(result)
    return result


def load_base():
    return VideoMAEForVideoClassification.from_pretrained(
        MODEL_ID, revision=REVISION, attn_implementation='sdpa')


def fine_tune(sequences, source, output_dir, epochs, samples, seed, device):
    config = {'pipeline_version': PIPELINE_VERSION, 'model_id': MODEL_ID, 'revision': REVISION,
              'epochs': epochs, 'samples_requested': samples, 'seed': seed, 'lr': 1e-5,
              'weight_decay': .01, 'unfrozen_blocks': 2, 'batch_size': 2,
              'clip_frames': CLIP_FRAMES, 'frame_step': FRAME_STEP, 'image_size': IMAGE_SIZE,
              'source': source, 'training_video': '0028_vid', 'held_out_video': '0031_vid'}
    folder = Path(output_dir) / 'encoders' / fingerprint(config)
    finished = folder / 'training.json'
    if finished.exists():
        print('Loading completed fine-tuned VideoMAE checkpoint', flush=True)
        model = VideoMAEForVideoClassification.from_pretrained(folder, attn_implementation='sdpa')
        return model.to(device).eval(), json.loads(finished.read_text()), folder
    folder.mkdir(parents=True, exist_ok=True)
    processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID, revision=REVISION)
    model = load_base()
    model.classifier = torch.nn.Linear(model.config.hidden_size, 2)
    model.config.num_labels = 2
    model.config.id2label = {0: 'background', 1: 'circling'}
    model.config.label2id = {'background': 0, 'circling': 1}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (model.videomae.encoder.layer[-2:], model.fc_norm, model.classifier):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    # A fingerprint verifies that training actually updates the encoder, not just its head.
    tracked = model.videomae.encoder.layer[-1].output.dense.weight
    initial = tracked.detach().clone()
    model.to(device)
    examples = training_examples(sequences, samples, seed)
    manifest = [{'video': sequences[i]['video'], 'sequence': sequences[i]['sequence'],
                 'center_frame': int(sequences[i]['frame_ids'][c]), 'label': label}
                for i, c, label in examples]
    pl.DataFrame(manifest).write_csv(folder / 'training_clips.csv')
    dataset = ClipDataset(sequences, examples, processor.image_mean, processor.image_std)
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=config['lr'], weight_decay=config['weight_decay'])
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = correct = count = 0
        for step, (pixels, labels) in enumerate(loader, 1):
            pixels, labels = pixels.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(pixel_values=pixels).logits
            loss = torch.nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite VideoMAE loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += int((logits.argmax(-1) == labels).sum())
            count += len(labels)
            if step % 16 == 0:
                print(f'VideoMAE epoch {epoch}/{epochs}, batch {step}/{len(loader)}, loss {total_loss/count:.4f}', flush=True)
        history.append({'epoch': epoch, 'loss': total_loss/count, 'train_accuracy': correct/count})
        print(f'VideoMAE epoch {epoch}: {history[-1]}', flush=True)
    delta = float((tracked.detach().cpu() - initial).abs().max())
    if delta == 0:
        raise RuntimeError('VideoMAE encoder did not change during fine-tuning')
    model.eval().cpu()
    model.save_pretrained(folder, safe_serialization=True)
    processor.save_pretrained(folder)
    info = {**config, 'device': device, 'history': history, 'training_clips': len(examples),
            'encoder_weight_max_change': delta, 'embedding_size': model.config.hidden_size,
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'adaptation': 'Supervised clip classification; last two encoder blocks, pooling norm and new binary head trained. '
                          'Earlier encoder layers frozen. No masked reconstruction pretraining on seed videos.'}
    write_json(finished, info)
    return model.to(device).eval(), info, folder


def extract_embeddings(model, sequences, encoder_folder, output_dir, stride, device):
    processor = VideoMAEImageProcessor.from_pretrained(encoder_folder)
    dtype = next(model.parameters()).dtype
    key = fingerprint({'encoder': encoder_folder.name, 'anchor_stride': stride, 'dtype': str(dtype)})
    folder = Path(output_dir) / 'embeddings' / key
    folder.mkdir(parents=True, exist_ok=True)
    results = []
    for i, sequence in enumerate(sequences):
        path = folder / f"{sequence['sequence']}.npz"
        # Include both excerpt endpoints in the interpolation domain (last cached frame
        # may precede the final source frame by one; endpoint extension is explicit).
        centers = np.unique(np.r_[np.arange(0, len(sequence['frame_ids']), stride // FRAME_STEP),
                                  len(sequence['frame_ids']) - 1])
        if not path.exists():
            examples = [(i, int(c), 0) for c in centers]
            dataset = ClipDataset(sequences, examples, processor.image_mean, processor.image_std)
            loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
            chunks = []
            with torch.inference_mode():
                for pixels, _ in loader:
                    hidden = model.videomae(pixel_values=pixels.to(device=device, dtype=dtype)).last_hidden_state
                    embeddings = model.fc_norm(hidden.mean(dim=1))
                    chunks.append(embeddings.float().cpu().numpy())
            embeddings = np.concatenate(chunks).astype(np.float32)
            if not np.isfinite(embeddings).all():
                raise ValueError('Nonfinite VideoMAE embeddings')
            partial = path.with_suffix('.partial.npz')
            np.savez_compressed(partial, frames=sequence['frame_ids'][centers], embeddings=embeddings)
            partial.replace(path)
        with np.load(path) as saved:
            anchors, embeddings = saved['frames'], saved['embeddings']
        frames = np.arange(sequence['start'], sequence['end'] + 1)
        dense = np.stack([np.interp(frames, anchors, embeddings[:, d])
                          for d in range(embeddings.shape[1])], axis=1).astype(np.float32)
        results.append({'video': sequence['video'], 'sequence': sequence['sequence'], 'frames': frames,
                        'x': dense, 'y': span_labels(frames, sequence['rows'])})
        print(f"Embeddings {i+1}/{len(sequences)}: {sequence['sequence']}, {len(anchors)} anchors", flush=True)
    return results, folder


def run_pipeline(epochs=30, context=150, videomae_epochs=3, samples=256, stride=16,
                 output_dir=OUTPUT, data_dir=ROOT/'data', span_dir=ROOT/'outputs'/'merged_spans', seed=42):
    if videomae_epochs < 1 or samples < 2 or samples % 2 or stride < 2 or stride % FRAME_STEP:
        raise ValueError('Use positive epochs, an even sample count and an even stride >=2')
    start = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    # MPS SDPA does not guarantee bitwise reproducibility; seed and device are recorded.
    torch.use_deterministic_algorithms(False)
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'VideoMAE device: {device}', flush=True)
    sequences, source = prepare_excerpts(data_dir, span_dir, output_dir, context)
    model, training, encoder_folder = fine_tune(sequences, source, output_dir, videomae_epochs, samples, seed, device)
    inference_dtype = torch.float16 if device in ('mps', 'cuda') else torch.float32
    model.to(dtype=inference_dtype)
    features, embedding_folder = extract_embeddings(model, sequences, encoder_folder, output_dir, stride, device)
    del model
    if device == 'mps':
        torch.mps.empty_cache()
    metadata = {'type': 'videomae', 'encoder': training, 'checkpoint': str(encoder_folder),
                'inference_dtype': str(inference_dtype),
                'versions': {name: version(name) for name in ('torch', 'transformers', 'safetensors')},
                'embedding_directory': str(embedding_folder), 'anchor_stride_frames': stride,
                'alignment': '16 RGB frames sampled every 2 source frames around each anchor; '
                    'edge replication within each excerpt. 768-D mean-pooled, normalized embeddings. '
                    'Linear interpolation between anchors to original frame resolution; no interpolation across gaps.',
                'spatial_preprocessing': 'Full scene letterboxed to 224x224 RGB, ImageNet mean/std; no center crop.',
                'evaluation_scope': 'All buffered merged-span excerpts, not full seed videos.'}
    if source_hash(span_dir) != source['span_sha256']:
        raise RuntimeError('Merged-span annotations changed during extraction; rerun to keep labels consistent')
    result = run_test(data_dir=data_dir, span_dir=span_dir, output_dir=output_dir, epochs=epochs,
                      context=context, seed=seed, feature_sequences=features,
                      feature_names=[f'videomae_{i:03d}' for i in range(768)], feature_source=metadata)
    result['elapsed_seconds'] = time.monotonic() - start
    write_json(output_dir / 'metrics.json', result)
    print(f'Complete in {result["elapsed_seconds"]:.1f} seconds: {output_dir}', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=30, help='MS-TCN epochs')
    parser.add_argument('--videomae-epochs', type=int, default=3)
    parser.add_argument('--samples', type=int, default=256, help='Balanced seed clips for encoder fine-tuning')
    parser.add_argument('--stride', type=int, default=16, help='Source frames between embedding anchors')
    parser.add_argument('--context', type=int, default=150)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = parser.parse_args()
    run_pipeline(**vars(args))

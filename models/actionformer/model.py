"""Compact one-class ActionFormer configuration and portable inference helpers."""
from copy import deepcopy

import numpy as np
import torch

from models.actionformer.vendor.libs.modeling import make_meta_arch
from models.actionformer.vendor.libs.utils import batched_nms

WINDOW = 256
DEFAULT_CONFIG = {
    'dataset': {
        'input_dim': 1, 'num_classes': 1, 'max_seq_len': WINDOW,
        'feat_stride': 1, 'num_frames': 0, 'default_fps': 1,
        'downsample_rate': 1, 'trunc_thresh': 0.5, 'crop_ratio': None,
        'force_upsampling': False,
    },
    'model': {
        'backbone_type': 'convTransformer', 'fpn_type': 'identity',
        'backbone_arch': (2, 2, 3), 'scale_factor': 2,
        'regression_range': [(0, 4), (4, 16), (16, 64), (64, 10000)],
        'n_head': 4, 'n_mha_win_size': 16, 'embd_kernel_size': 3,
        'embd_dim': 128, 'embd_with_ln': True, 'fpn_dim': 128,
        'fpn_with_ln': True, 'fpn_start_level': 0, 'head_dim': 128,
        'head_kernel_size': 3, 'head_num_layers': 3, 'head_with_ln': True,
        'max_buffer_len_factor': 6.0, 'use_abs_pe': False, 'use_rel_pe': False,
    },
    'train_cfg': {
        'center_sample': 'radius', 'center_sample_radius': 1.5,
        'loss_weight': 1.0, 'cls_prior_prob': 0.01, 'init_loss_norm': 100,
        'clip_grad_l2norm': 5.0, 'head_empty_cls': [], 'dropout': 0.1,
        'droppath': 0.1, 'label_smoothing': 0.0,
    },
    'test_cfg': {
        'pre_nms_thresh': 0.0001, 'pre_nms_topk': 1000, 'iou_threshold': 0.3,
        'min_score': 0.0, 'max_seg_num': 200, 'nms_method': 'soft',
        'nms_sigma': 0.5, 'duration_thresh': 0.0, 'multiclass_nms': True,
        'ext_score_file': None, 'voting_thresh': 0.75,
    },
}


def model_config(input_dim, **overrides):
    config = deepcopy(DEFAULT_CONFIG)
    config['dataset']['input_dim'] = int(input_dim)
    for key, value in overrides.items():
        if key in config and isinstance(value, dict):
            config[key].update(value)
        else:
            config[key] = value
    config['model']['input_dim'] = config['dataset']['input_dim']
    config['model']['num_classes'] = config['dataset']['num_classes']
    config['model']['max_seq_len'] = config['dataset']['max_seq_len']
    config['model']['train_cfg'] = config['train_cfg']
    config['model']['test_cfg'] = config['test_cfg']
    return config


def build_model(input_dim, **overrides):
    config = model_config(input_dim, **overrides)
    return make_meta_arch(config['model_name'] if 'model_name' in config else 'LocPointTransformer',
                          **config['model']), config


def video_item(features, start=0, video_id='video'):
    """Create the official one-video inference dictionary from a float array."""
    values = torch.as_tensor(np.asarray(features, dtype=np.float32).T)
    length = values.shape[1]
    return {
        'feats': values, 'video_id': video_id, 'fps': 1, 'duration': length,
        'feat_stride': 1, 'feat_num_frames': 0, 'window_start': int(start),
    }


def infer_windows(model, windows, device, min_score=0.01):
    """Run native ActionFormer decoding on a sequence of feature windows.

    Returns continuous global-frame segments and confidence scores. No frame
    threshold/span rule is involved.
    """
    was_training = model.training
    model.eval()
    results = []
    with torch.no_grad():
        for start, features in windows:
            item = video_item(features, start=start)
            output = model([item])[0]
            segments = output['segments'].cpu().numpy()
            scores = output['scores'].cpu().numpy()
            if len(segments):
                segments = segments + float(start)
                keep = scores >= min_score
                results.extend((tuple(segment), float(score))
                               for segment, score in zip(segments[keep], scores[keep]))
    if was_training:
        model.train()
    return results


def suppress_cross_window_duplicates(proposals, iou_threshold=0.3, max_segments=1000):
    """Apply class-agnostic hard NMS to proposals from overlapping windows."""
    if not proposals:
        return []
    segments = torch.as_tensor([item[0] for item in proposals], dtype=torch.float32)
    scores = torch.as_tensor([item[1] for item in proposals], dtype=torch.float32)
    labels = torch.zeros(len(proposals), dtype=torch.long)
    segments, scores, _ = batched_nms(
        segments, scores, labels, iou_threshold=iou_threshold, min_score=0,
        max_seg_num=max_segments, use_soft_nms=False, multiclass=False,
        voting_thresh=0,
    )
    return [(tuple(segment), float(score)) for segment, score in zip(segments.numpy(), scores.numpy())]

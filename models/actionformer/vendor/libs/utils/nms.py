"""Portable PyTorch 1-D NMS for the vendored ActionFormer model.

The upstream implementation calls its compiled ``nms_1d_cpu`` extension. This
adaptation preserves hard NMS, Gaussian Soft-NMS, optional voting and the same
public ``batched_nms`` interface without a platform-specific binary.
"""
import torch


def _iou_one(segment, segments):
    left = torch.maximum(segment[0], segments[:, 0])
    right = torch.minimum(segment[1], segments[:, 1])
    intersection = (right - left).clamp(min=0)
    union = ((segment[1] - segment[0]).clamp(min=0)
             + (segments[:, 1] - segments[:, 0]).clamp(min=0) - intersection)
    return intersection / union.clamp(min=torch.finfo(segments.dtype).eps)


def _hard_nms(segs, scores, iou_threshold, min_score, max_num):
    keep = scores > min_score
    original_indices = torch.nonzero(keep, as_tuple=False).squeeze(dim=1)
    segs, scores = segs[keep], scores[keep]
    order = torch.argsort(scores, descending=True)
    selected = []
    while order.numel():
        index = order[0]
        selected.append(index)
        if max_num > 0 and len(selected) >= max_num:
            break
        if order.numel() == 1:
            break
        rest = order[1:]
        order = rest[_iou_one(segs[index], segs[rest]) <= iou_threshold]
    selected = torch.as_tensor(selected, dtype=torch.long, device=segs.device)
    return original_indices[selected]


def _soft_nms(segs, scores, iou_threshold, sigma, min_score, max_num):
    """Gaussian Soft-NMS (method=2 in the upstream extension)."""
    keep = scores > min_score
    segs, scores = segs[keep].clone(), scores[keep].clone()
    out_segs, out_scores = [], []
    while segs.shape[0] and (max_num <= 0 or len(out_segs) < max_num):
        index = int(torch.argmax(scores))
        best_seg, best_score = segs[index].clone(), scores[index].clone()
        out_segs.append(best_seg)
        out_scores.append(best_score.reshape(1))
        keep = torch.ones(segs.shape[0], dtype=torch.bool, device=segs.device)
        keep[index] = False
        segs, scores = segs[keep], scores[keep]
        if not segs.shape[0]:
            break
        iou = _iou_one(best_seg, segs)
        scores = scores * torch.exp(-(iou * iou) / max(float(sigma), 1e-6))
        keep = scores > min_score
        segs, scores = segs[keep], scores[keep]
    if not out_segs:
        return segs.new_empty((0, 2)), scores.new_empty((0,))
    return torch.stack(out_segs), torch.cat(out_scores)


def seg_voting(nms_segs, all_segs, all_scores, iou_threshold, score_offset=1.5):
    if not nms_segs.shape[0]:
        return nms_segs
    offset_scores = all_scores + score_offset
    left = torch.maximum(nms_segs[:, None, 0], all_segs[None, :, 0])
    right = torch.minimum(nms_segs[:, None, 1], all_segs[None, :, 1])
    intersection = (right - left).clamp(min=0)
    nms_lengths = (nms_segs[:, 1] - nms_segs[:, 0])[:, None]
    all_lengths = (all_segs[:, 1] - all_segs[:, 0])[None, :]
    iou = intersection / (nms_lengths + all_lengths - intersection).clamp(min=1e-6)
    weights = (iou >= iou_threshold).to(all_scores.dtype) * offset_scores[None, :] * iou
    weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
    return weights @ all_segs


def batched_nms(segs, scores, cls_idxs, iou_threshold, min_score, max_seg_num,
                use_soft_nms=True, multiclass=True, sigma=0.5, voting_thresh=0.75):
    """Class-aware or class-agnostic NMS, sorted by descending confidence."""
    if not segs.shape[0]:
        return segs.new_empty((0, 2)), scores.new_empty((0,)), cls_idxs.new_empty((0,))
    output_segs, output_scores, output_labels = [], [], []
    groups = torch.unique(cls_idxs) if multiclass else [None]
    for class_id in groups:
        indices = torch.arange(segs.shape[0], device=segs.device) if class_id is None else torch.where(cls_idxs == class_id)[0]
        current_segs, current_scores = segs[indices].cpu(), scores[indices].cpu()
        current_labels = cls_idxs[indices].cpu()
        if use_soft_nms:
            kept_segs, kept_scores = _soft_nms(current_segs, current_scores, iou_threshold,
                                               sigma, min_score, max_seg_num)
            # The soft-NMS loop returns confidence-sorted detections directly.
            kept_labels = current_labels.new_full((kept_segs.shape[0],),
                                                   int(class_id) if class_id is not None else 0)
        else:
            kept = _hard_nms(current_segs, current_scores, iou_threshold, min_score, max_seg_num)
            kept_segs, kept_scores, kept_labels = current_segs[kept], current_scores[kept], current_labels[kept]
        if not multiclass and voting_thresh > 0 and kept_segs.shape[0]:
            kept_segs = seg_voting(kept_segs, current_segs, current_scores, voting_thresh)
        output_segs.append(kept_segs)
        output_scores.append(kept_scores)
        output_labels.append(kept_labels)
    new_segs = torch.cat(output_segs)
    new_scores = torch.cat(output_scores)
    new_labels = torch.cat(output_labels)
    order = torch.argsort(new_scores, descending=True)
    if max_seg_num > 0:
        order = order[:max_seg_num]
    return new_segs[order], new_scores[order], new_labels[order]

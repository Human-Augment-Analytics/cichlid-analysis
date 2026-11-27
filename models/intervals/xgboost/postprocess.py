"""Centered majority voting on contiguous per-frame predictions."""

import numpy as np

VOTE_WINDOW = 30
VOTE_FRACTION = 0.5


def majority_vote_fraction(scores, threshold=0.5):
    """Fraction of positive predictions in each centered 30-frame window.

    The even window covers offsets -15..+14. At the beginning/end of a video,
    use only available frames. Ties are positive (at least 15/30 in full windows).
    Call once per full video, after joining inference batches, to preserve context.
    """
    scores = np.asarray(scores)
    if scores.ndim != 1:
        raise ValueError("Expected one score per contiguous video frame")
    centers = np.arange(len(scores))
    starts = np.maximum(0, centers - VOTE_WINDOW // 2)
    ends = np.minimum(len(scores), centers + VOTE_WINDOW // 2)
    cumulative = np.r_[0, np.cumsum(scores >= threshold, dtype=np.int64)]
    return (cumulative[ends] - cumulative[starts]) / (ends - starts)

"""Rule-based event proposer + direction scorer (Step 3).

Operates on the per-frame flow-score DataFrame produced by `flow.process_clip`.

Detection logic:
  1. Within a rolling window (default 2 s) compute the mean of sign(rot_omega),
     gated on activity > threshold. This gives a per-frame "sign-consistency"
     signal in [-1, +1].
  2. Threshold at |mean_sign| > sign_thr.  Positive (>+thr) runs are CW
     candidates; negative (<-thr) runs are CCW candidates.
  3. Merge gaps < merge_gap_s within each direction channel.
  4. Keep runs >= min_dur_s.

Each returned event has a score = mean(|rot_omega|) * duration_frames, meant to
be used for ranking (mAP) rather than for an absolute probability.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class Event:
    start_frame: int
    end_frame: int
    direction: str          # "CW" | "CCW"
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


# -----------------------------------------------------------------------------
# Rolling sign-consistency signal (also useful for plotting/debug)
# -----------------------------------------------------------------------------

def rolling_mean_sign(df_flow: pd.DataFrame, win: int,
                      activity_thr: float = 300) -> np.ndarray:
    """Mean of sign(rot_omega) in a rolling window, NaN where activity too low."""
    sg = np.sign(df_flow['rot_omega'].values).astype(float)
    sg[df_flow['activity'].values < activity_thr] = np.nan
    return pd.Series(sg).rolling(win, center=True, min_periods=5).mean().values


# -----------------------------------------------------------------------------
# Event proposal
# -----------------------------------------------------------------------------

def _runs_of_true(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return [(start, end_exclusive), ...] runs of True in mask."""
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == 0:
        return []
    padded = np.r_[False, mask, False]
    d = np.diff(padded.astype(int))
    starts = np.where(d == 1)[0]
    ends   = np.where(d == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def _merge_short_gaps(runs: list[tuple[int, int]],
                      max_gap: int) -> list[tuple[int, int]]:
    """Merge consecutive runs separated by < max_gap frames."""
    if not runs:
        return []
    out = [runs[0]]
    for s, e in runs[1:]:
        ps, pe = out[-1]
        if s - pe < max_gap:
            out[-1] = (ps, e)
        else:
            out.append((s, e))
    return out


def detect_events(df_flow: pd.DataFrame,
                  fps: float = 30.0,
                  sign_win_s: float = 2.0,
                  sign_thr: float = 0.6,
                  min_dur_s: float = 3.0,
                  merge_gap_s: float = 2.0,
                  activity_thr: float = 300,
                  min_score: float = 0.0,
                  max_centroid_spread_px: float | None = None,
                  min_centroid_frames: int = 10,
                  activity_area_range: tuple[float, float] | None = None,
                  centroid_metric: str = 'std',
                  smooth_centroid_win: int = 1) -> list[Event]:
    """Propose circling events from a flow-score DataFrame.

    Returned events use the *original* frame indices from `df_flow['frame']`.

    Filters (post-run):
    - `min_score`: drops events whose event-score < threshold.
    - `max_centroid_spread_px`: drops events whose activity-centroid spread
      exceeds the threshold. Spread metric controlled by `centroid_metric`:
        'std'      = sqrt(std(cx)^2 + std(cy)^2)                           [default]
        'endpoint' = ||centroid[end_valid] - centroid[start_valid]||        (Signal-B)
        'iqr'      = sqrt((p75-p25)^2 of cx + (p75-p25)^2 of cy)             (Signal-B)
      `smooth_centroid_win` > 1 applies a centred rolling mean to cx/cy
      before computing the metric (also Signal-B).
    - `min_centroid_frames`: minimum frames with valid centroid required to
      apply the spread check (otherwise event kept).
    - `activity_area_range`: (min, max) tuple. Drops events whose MEDIAN
      activity (active-pixel count) over the event falls outside this range.
      Rationale: pair circling has ~2 fish-bodies of motion; single fish turn
      has 1 body worth; multi-fish scatter has >>2 bodies.
    """
    sign_win  = max(1, int(sign_win_s  * fps))
    min_dur   = max(1, int(min_dur_s   * fps))
    merge_gap = max(1, int(merge_gap_s * fps))

    om  = df_flow['rot_omega'].values
    frm = df_flow['frame'].values.astype(int)
    mean_sign = rolling_mean_sign(df_flow, sign_win, activity_thr)

    events: list[Event] = []
    for direction, mask in (('CW',  mean_sign >  sign_thr),
                            ('CCW', mean_sign < -sign_thr)):
        runs = _merge_short_gaps(_runs_of_true(np.nan_to_num(mask, nan=0) > 0.5),
                                 merge_gap)
        for s, e in runs:
            if e - s < min_dur:
                continue
            start_f = int(frm[s])
            end_f   = int(frm[min(e - 1, len(frm) - 1)])
            seg_om  = om[s:e]
            score   = float(np.abs(seg_om).mean() * (e - s))
            if score < min_score:
                continue
            seg_df = df_flow.iloc[s:e]
            if activity_area_range is not None:
                med_act = float(seg_df['activity'].median())
                lo, hi = activity_area_range
                if med_act < lo or med_act > hi:
                    continue
            if max_centroid_spread_px is not None:
                cx = seg_df['cx'].dropna().values
                cy = seg_df['cy'].dropna().values
                if len(cx) >= min_centroid_frames:
                    if smooth_centroid_win > 1:
                        cxs = pd.Series(cx).rolling(smooth_centroid_win, center=True,
                                                    min_periods=1).mean().values
                        cys = pd.Series(cy).rolling(smooth_centroid_win, center=True,
                                                    min_periods=1).mean().values
                    else:
                        cxs, cys = cx, cy
                    if centroid_metric == 'endpoint':
                        spread = float(np.hypot(cxs[-1] - cxs[0], cys[-1] - cys[0]))
                    elif centroid_metric == 'iqr':
                        spread = float(np.sqrt(
                            (np.percentile(cxs, 75) - np.percentile(cxs, 25)) ** 2 +
                            (np.percentile(cys, 75) - np.percentile(cys, 25)) ** 2))
                    else:  # 'std'
                        spread = float(np.sqrt(cxs.std() ** 2 + cys.std() ** 2))
                    if spread > max_centroid_spread_px:
                        continue
            events.append(Event(start_f, end_f, direction, score))

    events.sort(key=lambda ev: (ev.start_frame, ev.end_frame))
    return events


# -----------------------------------------------------------------------------
# Direction prediction on a given (GT or predicted) span
# -----------------------------------------------------------------------------

def num_turns_on_span(df_flow: pd.DataFrame,
                      start_frame: int,
                      end_frame: int,
                      activity_thr: float = 300,
                      round_to: float = 0.5) -> float | None:
    """Predict number of turns for a given span.

    Turns = |integral of omega over span| / (2 pi), where omega = rot_omega
    (angular-velocity-like signal in rad/frame).

    Returns the absolute number of turns (direction is a separate output).
    Rounds to the nearest `round_to` (GT annotations are in 0.5 increments).
    Returns None if the span has too few active frames to judge.
    """
    frm = df_flow['frame'].values.astype(int)
    mask = (frm >= start_frame) & (frm <= end_frame)
    sub = df_flow[mask]
    if len(sub) == 0:
        return None
    sub = sub[sub['activity'] >= activity_thr]
    if len(sub) < 10:
        return None
    total_rad = float(sub['rot_omega'].sum())
    turns = abs(total_rad) / (2.0 * np.pi)
    if round_to and round_to > 0:
        turns = round(turns / round_to) * round_to
    return float(turns)


def direction_on_span(df_flow: pd.DataFrame,
                      start_frame: int,
                      end_frame: int,
                      activity_thr: float = 50,
                      min_valid_frames: int = 10) -> str | None:
    """Predict "CW" or "CCW" for a span via sign-majority of rot_omega.

    For each frame with activity >= `activity_thr`, we take the sign of the
    per-frame angular-velocity estimate rot_omega. The returned direction is
    whichever sign appears more often across the span (plurality vote).

    Tie-break: when the CW vs CCW vote counts are exactly equal, fall back
    to the sign of the signed mean rot_omega across the same frames. A
    tie is rare on real events (even a 1-frame excess breaks it), but this
    still gives a deterministic answer whenever any signed rotation signal
    exists. Returns None only if (a) fewer than `min_valid_frames` scorable
    frames remain, or (b) the vote is tied AND the signed mean is exactly 0.

    Design history:
      - Started with mean(rot_orb) -- unbounded signal, hijacked by handfuls
        of high-activity outlier frames (see vid0002 evt_0048 incident).
      - Switched to mean(rot_omega) -- bounded signal, robust to the big
        outliers, but still prone to flipping when a handful of frames with
        moderately large rot_omega outweigh a larger count of frames with
        smaller (correct-signed) rot_omega (see vid0031 evt_0023 and
        vid0037 evt_0066).
      - Sign-majority removes magnitude from the decision entirely. Noise
        frames add symmetric +/- votes that cancel; real rotation produces
        an asymmetric vote. Matches the physical intuition: "which way is
        the pair rotating for most of the event?"

    The default `activity_thr=50` gives broad frame coverage without
    admitting pure sensor-noise frames (empirically, pair motion produces
    activity >= 50 over most of the event even on quieter tanks).
    """
    frm = df_flow['frame'].values.astype(int)
    mask = (frm >= start_frame) & (frm <= end_frame)
    sub = df_flow[mask]
    if len(sub) == 0:
        return None
    sub = sub[sub['activity'] >= activity_thr]
    if len(sub) < min_valid_frames:
        return None
    pos = int((sub['rot_omega'] > 0).sum())
    neg = int((sub['rot_omega'] < 0).sum())
    if pos != neg:
        return 'CW' if pos > neg else 'CCW'
    # Tie-break: fall back to the sign of the signed mean rot_omega.
    m = float(sub['rot_omega'].mean())
    if m == 0:
        return None
    return 'CW' if m > 0 else 'CCW'


# -----------------------------------------------------------------------------
# Helpers for converting between clip-local and video-global frames
# -----------------------------------------------------------------------------

def shift_events(events: Iterable[Event], offset: int) -> list[Event]:
    """Return events with start/end frames offset by `offset`."""
    return [Event(e.start_frame + offset, e.end_frame + offset,
                  e.direction, e.score) for e in events]

import json
from pathlib import Path
import numpy as np
import pandas as pd


def load_table(path):
    return pd.read_parquet(path) if Path(path).suffix == '.parquet' else pd.read_csv(path)


def normalize_pose(df, mapping, start, end):
    """Adapt wide pose rows into long keypoint rows; retain only the review window."""
    frame, track = mapping['frame'], mapping['track']
    missing = {frame, track} - set(df.columns)
    if missing:
        raise ValueError(f'Missing pose columns: {missing}. Available: {list(df.columns)}')
    df = df[df[frame].between(start, end)].copy()
    out = []
    for name, spec in mapping['keypoints'].items():
        cols = [spec['x'], spec['y']] + ([spec['confidence']] if spec.get('confidence') else [])
        if set(cols) - set(df.columns):
            raise ValueError(f'Missing columns for {name}: {cols}')
        r = pd.DataFrame({'frame': df[frame], 'track': df[track].astype(str),
                          'keypoint': name, 'x': df[spec['x']], 'y': df[spec['y']],
                          'confidence': df[spec['confidence']] if spec.get('confidence') else 1.0})
        out.append(r)
    if not out:
        raise ValueError('Configure at least one keypoint')
    rows = pd.concat(out, ignore_index=True)
    if rows.duplicated(['frame', 'track', 'keypoint']).any():
        raise ValueError('Duplicate frame/track/keypoint rows; resolve multiple detections first')
    rows['frame'] = pd.to_numeric(rows['frame'], errors='raise')
    if not np.isfinite(rows['frame']).all() or (rows['frame'] % 1 != 0).any():
        raise ValueError('Frames must be finite zero-based integers')
    for c in ['x', 'y', 'confidence']:
        rows[c] = pd.to_numeric(rows[c], errors='coerce')
    return rows


def features(rows, pair, start, end, fps, threshold=.5, centroid_keypoints=None):
    """Pair bearing rate in rad/s. No derivatives across missing frames."""
    good = rows[(rows.confidence >= threshold) & np.isfinite(rows.x) & np.isfinite(rows.y)]
    if centroid_keypoints:
        good = good[good.keypoint.isin(centroid_keypoints)]
    centers = good.groupby(['frame', 'track'])[['x', 'y']].mean()
    frames = np.arange(start, end + 1)
    xy = []
    for fish in pair:
        try:
            xy.append(centers.xs(str(fish), level='track').reindex(frames).to_numpy())
        except KeyError:
            xy.append(np.full((len(frames), 2), np.nan))
    rel = xy[1] - xy[0]
    distance = np.linalg.norm(rel, axis=1)
    angle = np.arctan2(rel[:, 1], rel[:, 0])
    angle[distance < 1e-8] = np.nan
    delta = np.arctan2(np.sin(np.diff(angle)), np.cos(np.diff(angle)))
    rate = np.r_[np.nan, delta * fps]
    # Canvas/image y grows downward: positive bearing rate is clockwise visually.
    return pd.DataFrame({'frame': frames, 'distance_px': distance,
                         'bearing_rad': angle, 'angular_velocity_rad_s': rate})


def annotations(df, start, end, video_id=None):
    """Canonical inclusive endpoints; unknown remains unknown."""
    if video_id is not None and 'video_id' in df:
        df = df[df.video_id.astype(str) == str(video_id)]
    result = []
    for row in df.to_dict('records'):
        a, b = int(row['start_frame']), int(row['end_frame'])
        if a > b:
            raise ValueError('Annotation start exceeds end')
        if a > end or b < start:
            continue
        verified = str(row.get('Verified', row.get('verified', ''))).lower()
        status = 'circling' if verified in ('true', '1', '1.0') else 'unknown'
        if str(row.get('behavior', '')).lower() == 'non-circling' and verified in ('true','1','1.0'):
            status = 'non-circling'
        result.append({'start': max(a, start), 'end': min(b, end), 'status': status,
                       'notes': str(row.get('notes', '')), 'event': str(row.get('event_id', ''))})
    return result


def payload(rows, feat, spans, scores, meta):
    tracks = {}
    for f, group in rows.groupby('frame'):
        tracks[str(int(f))] = group[['track','keypoint','x','y','confidence']].to_dict('records')
    obj = dict(meta, poses=tracks, features=feat.to_dict('records'), spans=spans, scores=scores)
    # Strict JSON: browsers cannot parse Python's NaN literals.
    def clean(x):
        if isinstance(x, dict): return {str(k): clean(v) for k,v in x.items()}
        if isinstance(x, list): return [clean(v) for v in x]
        if isinstance(x, (float, np.floating)): return float(x) if np.isfinite(x) else None
        if isinstance(x, np.integer): return int(x)
        return x
    return clean(obj)


def write_viewer(out, data):
    template = Path(__file__).with_name('viewer.html').read_text()
    text = json.dumps(data, allow_nan=False).replace('<', '\\u003c')
    (Path(out)/'index.html').write_text(template.replace('__DATA__', text))


def pair_table(df, mapping, pair, start, end, frame_index=None):
    """Read already engineered numeric features for an explicitly selected ordered pair."""
    framecol=mapping['frame'];names=mapping['features']
    if not names or len(set(names))!=len(names):raise ValueError('Configure unique nonempty pair feature column names')
    needed=[framecol]+names
    if mapping.get('fish_a') or mapping.get('fish_b'):
        if not (mapping.get('fish_a') and mapping.get('fish_b')):raise ValueError('Configure both pair ID columns')
        needed += [mapping['fish_a'],mapping['fish_b']]
    missing=set(needed)-set(df.columns)
    if missing:raise ValueError(f'Missing pair columns {missing}. Available: {list(df.columns)}')
    df=df.copy();df[framecol]=pd.to_numeric(df[framecol],errors='raise')
    if not np.isfinite(df[framecol]).all() or (df[framecol]%1!=0).any() or (df[framecol]<0).any():
        raise ValueError('Pair frames must be nonnegative finite integers')
    if mapping.get('fish_a'):
        df=df[(df[mapping['fish_a']].astype(str)==str(pair[0]))&(df[mapping['fish_b']].astype(str)==str(pair[1]))]
    df=df[df[framecol].between(start,end)]
    if df[framecol].duplicated().any():raise ValueError('Multiple pair rows per frame; configure fish_a/fish_b IDs or aggregate upstream explicitly')
    if not len(df):raise ValueError('No rows match the selected pair/window')
    frames=np.arange(start,end+1) if frame_index is None else np.asarray(frame_index);table=df.set_index(framecol).reindex(frames)
    for name in names:table[name]=pd.to_numeric(table[name],errors='raise')
    x=table[names].to_numpy(dtype=np.float32);x[~np.isfinite(x)]=np.nan
    valid=np.isfinite(x).any(axis=1)
    feat=pd.DataFrame({'frame':frames,'distance_px':np.nan,'bearing_rad':np.nan,'angular_velocity_rad_s':np.nan})
    for target,key in [('distance_px','distance'),('angular_velocity_rad_s','angular_velocity'),('bearing_rad','bearing')]:
        if mapping.get(key):
            if mapping[key] not in names:raise ValueError(f'{key} plot column must also be listed in features')
            feat[target]=pd.to_numeric(table[mapping[key]],errors='raise').to_numpy()
    return x,valid,feat

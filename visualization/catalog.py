"""Recording catalog and on-demand pose, score and feature adapters."""
from collections import OrderedDict
import ast
import json
import math
from pathlib import Path
import re
import subprocess
import threading
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from .live_intervals import load_intervals, KEYPOINTS, MIDLINE

MODEL_NAMES = {'rule_based': 'Rule-based detector', 'logistic_regression': 'Logistic regression', 'circling_baseline': 'Logistic regression', 'xgboost': 'XGBoost', 'mstcn_dinov2': 'MS-TCN + DINOv2'}

FEATURES = {
    'centroid_distance_px': 'Shared centroid distance (px)',
    'centroid_distance_bl': 'Shared centroid distance (body lengths)',
    'd_orbital_rad_s': 'Shared orbital angular velocity (rad/s)',
    'd_orbital_rad_s_clean': 'Cleaned orbital angular velocity (rad/s)',
    'w4s_close_frac': 'Close fraction · 4 s',
    'w4s_orbit_turns_abs': 'Absolute orbital turns · 4 s',
    'w4s_orbit_turns_net': 'Net orbital turns · 4 s',
    'mutual_facing': 'Mutual facing',
    'speed_mean_bl_s': 'Mean pair speed (body lengths/s)',
    'closing_speed_bl_s': 'Closing speed (body lengths/s)',
}

def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, list): return [clean(v) for v in x]
    if isinstance(x, (float, np.floating)): return float(x) if np.isfinite(x) else None
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, np.bool_): return bool(x)
    return x

def json_bytes(x): return json.dumps(clean(x), allow_nan=False).encode()

def source_url(url):
    """Only URLs from the local trusted catalog are accepted, never from HTTP parameters."""
    parts = urlsplit(str(url))
    if parts.scheme != 'https' or not parts.hostname or parts.username or parts.password:
        raise ValueError('Remote video must be a public HTTPS URL without embedded credentials')
    if parts.hostname in {'dropbox.com', 'www.dropbox.com'}:
        query = [(k, v) for k, v in parse_qsl(parts.query) if k not in {'dl', 'raw'}]
        query.append(('raw', '1'))
        return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ''))
    return url

def read_urls(repo):
    """Read the repo's known URL dictionary without importing OpenCV/model dependencies."""
    p = repo / 'video_source.py'
    if not p.is_file(): return {}
    try:
        for node in ast.parse(p.read_text()).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'VIDEO_URLS' for t in node.targets):
                return ast.literal_eval(node.value)
    except (SyntaxError, ValueError): pass
    return {}

def fields(path): return pq.ParquetFile(path).schema_arrow.names

def scoped(path, start, end, video_id, expected=None, columns=None):
    """Frame predicates first; video and project predicates prevent mixed-table collisions."""
    names = fields(path); filters = [('FrameNum', '>=', max(0, start)), ('FrameNum', '<=', end)]
    vc = next((c for c in ('day_label', 'VideoID', 'video_id', 'video') if c in names), None)
    if vc: filters.append((vc, '==', video_id))
    if expected and 'project_id' in names: filters.append(('project_id', '==', expected))
    wanted = None if columns is None else list(dict.fromkeys([c for c in columns + [vc, 'project_id'] if c in names]))
    return pd.read_parquet(path, filters=filters, columns=wanted)

class Catalog:
    def __init__(self, path):
        self.path = Path(path).resolve(); raw = json.loads(self.path.read_text()); self.base = self.path.parent
        self.repo = (self.base / raw.get('repo_root', '..')).resolve()
        self.entries = {}; self.cache = OrderedDict(); self.lock = threading.Lock()
        self.single = 'recordings' not in raw
        entries = [raw] if self.single else raw['recordings']
        urls = read_urls(self.repo)
        for entry in entries:
            entry = dict(entry); key = entry.get('id', entry['video_id'])
            if key in self.entries: raise ValueError(f'Duplicate catalog ID: {key}')
            entry['id'] = key
            if not entry.get('video_url'): entry['video_url'] = urls.get(Path(entry.get('video', entry['video_id'] + '.mp4')).name)
            self.entries[key] = entry
        if raw.get('discover_local', False):
            directory = self.resolve(raw.get('data_dir', '../data'))
            for video in sorted(directory.glob('*_vid.mp4')):
                if any(self.resolve(e['video']) == video.resolve() for e in self.entries.values() if e.get('video')): continue
                vid = video.stem; key = 'local:' + vid
                self.entries[key] = dict(id=key, video_id=vid, label=vid + ' · local recording (confirm identity)', video=str(video), poses=str(video.with_suffix('.parquet')), fps=30, identity_unconfirmed=True)
        self.default = raw.get('default_recording', next(iter(self.entries), None))
        if not self.entries: raise ValueError('Catalog contains no recordings')
    def resolve(self, value): return (self.base / value).resolve()
    def get(self, key=None):
        key = key or self.default
        if key not in self.entries: raise ValueError('Unknown recording ID')
        with self.lock:
            if key not in self.cache:
                self.cache[key] = Recording(self, self.entries[key])
                while len(self.cache) > 3: self.cache.popitem(last=False)
            self.cache.move_to_end(key)
            return self.cache[key]
    def listing(self):
        return dict(default=self.default, recordings=[self.availability(e) for e in self.entries.values()])
    def availability(self, e):
        local = bool(e.get('video') and self.resolve(e['video']).is_file())
        poses = bool(e.get('poses') and self.resolve(e['poses']).is_file())
        remote = bool(e.get('video_url'))
        pred = list((self.repo / 'outputs/predictions').glob('*/' + e['video_id'] + '.csv')) if e.get('discover_predictions', False) else []
        pred += [self.resolve(s['path']) for s in e.get('tracks', []) if s.get('kind') == 'prediction' and self.resolve(s['path']).is_file()]
        annotation = any(self.resolve(s['path']).is_file() for s in e.get('tracks', []) if s.get('kind', 'annotation') == 'annotation')
        return dict(id=e['id'], video_id=e['video_id'], label=e.get('label', e['video_id']), local=local, dropbox=remote, poses=poses, predictions=len(set(pred)), annotations=annotation)

class Recording:
    def __init__(self, catalog, cfg):
        self.catalog = catalog; self.cfg = cfg; self.resolve = catalog.resolve
        self.video_id = cfg['video_id']; self.id = cfg['id']; self.fps = float(cfg.get('fps', 30))
        if not math.isfinite(self.fps) or self.fps <= 0: raise ValueError('FPS must be positive')
        self.video = self.resolve(cfg['video']) if cfg.get('video') else None
        self.remote = source_url(cfg['video_url']) if cfg.get('video_url') else None
        self.warnings = []; self.chunks = OrderedDict(); self.lock = threading.Lock(); self.probe = None
        if cfg.get('identity_unconfirmed'): self.warnings.append('Locally discovered recording: verify project, video and pose identity before interpreting results. FPS defaults to 30; edit the catalog if different.')
        self.layers = {}; self.keys = {}
        for name, key in [('raw', 'poses'), ('cleaned', 'cleaned_poses')]:
            p = self.resolve(cfg[key]) if cfg.get(key) else None
            if not p or not p.is_file(): continue
            try:
                names = fields(p)
                metadata = pq.ParquetFile(p).schema_arrow.metadata or {}
                for field, expected in [('day_label', self.video_id), ('project_id', cfg.get('expected_project_id'))]:
                    if expected and field.encode() in metadata and metadata[field.encode()].decode() != expected:
                        raise ValueError(f'{field} metadata does not match this recording')
                if {'FrameNum', 'TrackUID', 'X_center', 'Y_center'} - set(names): raise ValueError('Missing required pose columns')
                self.layers[name] = p; self.keys[name] = [k for k in KEYPOINTS if k+'_x' in names and k+'_y' in names]
            except Exception as exc: self.warnings.append(f'{name} pose input rejected: {exc}')
        if not self.layers: self.warnings.append('Pose Parquet missing: video-only playback; no skeletons or movement geometry.')
        self.tracks = []; self.score_specs = []; seen = set()
        specs = list(cfg.get('tracks', []))
        if cfg.get('discover_predictions', False):
            specs += [dict(name=MODEL_NAMES.get(p.parent.name, p.parent.name.replace('_', ' ')), kind='prediction', path=str(p)) for p in sorted((catalog.repo / 'outputs/predictions').glob('*/' + self.video_id + '.csv'))]
        for spec in specs:
            p = self.resolve(spec['path'])
            if p in seen: continue
            seen.add(p)
            if not p.is_file():
                self.warnings.append(f"Optional interval source missing: {spec.get('name', p.name)}"); continue
            try:
                spans, skipped = load_intervals(p, self.video_id, self.fps, spec)
                self.tracks.append(dict(name=spec.get('name', p.parent.name), kind=spec.get('kind', 'annotation'), spans=spans, skipped=skipped, source=str(p.relative_to(catalog.repo)) if p.is_relative_to(catalog.repo) else p.name))
                if skipped: self.warnings.append(f'{p.name}: skipped {skipped} incomplete annotation rows')
            except Exception as exc: self.warnings.append(f'{p.name} interval source rejected: {exc}')
        scores = cfg.get('scores', [])
        scores = [scores] if isinstance(scores, dict) else list(scores)
        if cfg.get('discover_predictions', False):
            for p in sorted((catalog.repo / 'outputs').glob('*/scores/' + self.video_id + '_scores.parquet')):
                scores += [dict(name=MODEL_NAMES.get(p.parent.parent.name, p.parent.parent.name.replace('_', ' ')), path=str(p), frame='FrameNum', score='circling_score')]
        seen = set()
        for spec in scores:
            p = self.resolve(spec['path'])
            if p in seen: continue
            seen.add(p)
            if not p.is_file(): continue
            try:
                names = fields(p) if p.suffix == '.parquet' else list(pd.read_csv(p, nrows=0))
                fc = spec.get('frame', 'FrameNum' if 'FrameNum' in names else 'frame'); sc = spec.get('score', 'circling_score' if 'circling_score' in names else 'score')
                if fc not in names or sc not in names: raise ValueError('Score/frame columns not found')
                self.score_specs.append(dict(name=spec.get('name', p.parent.parent.name), path=p, frame=fc, score=sc, video=spec.get('video'), index='score'+str(len(self.score_specs))))
            except Exception as exc: self.warnings.append(f'{p.name} scores rejected: {exc}')
        self.feature_path = self.resolve(cfg['features']) if cfg.get('features') else catalog.repo / 'outputs/pair_motion/pair_features_frame.parquet'
        self.feature_names = []
        if (cfg.get('features') or cfg.get('discover_predictions', False)) and self.feature_path.is_file():
            names = fields(self.feature_path)
            if {'FrameNum', 'TrackUID_1', 'TrackUID_2'} <= set(names):
                if any(c in names for c in ('day_label', 'VideoID', 'video_id', 'video')) or cfg.get('features'):
                    self.feature_names = [c for c in FEATURES if c in names]
        self.meta_base = dict(id=self.id, video_id=self.video_id, fps=self.fps, start_frame=int(cfg.get('start_frame', 0)), width=int(cfg.get('width', 1296)), height=int(cfg.get('height', 972)), duration=float(cfg.get('duration', cfg.get('frame_count', 1)/self.fps)), chunk_frames=round(self.fps*10), layers=list(self.layers), keys=self.keys, skeleton=list(zip(MIDLINE[:-1], MIDLINE[1:])), tracks=self.tracks, scores=[dict(id=s['index'], name=s['name'], source=str(s['path'].relative_to(catalog.repo)) if s['path'].is_relative_to(catalog.repo) else s['path'].name) for s in self.score_specs], features=[dict(id=n, name=FEATURES[n]) for n in self.feature_names], warnings=self.warnings)
    def media(self, mode):
        if mode not in {'auto', 'local', 'dropbox'}: raise ValueError('Unknown source mode')
        has_local = bool(self.video and self.video.is_file())
        if mode == 'local' and not has_local: raise ValueError('Local video missing. Choose Dropbox/Auto or download it into data/.')
        if mode == 'dropbox' and not self.remote: raise ValueError('No remote video URL configured for this recording')
        actual = 'local' if has_local and mode != 'dropbox' else 'dropbox'
        if actual == 'dropbox' and not self.remote: raise ValueError('No video source available')
        return actual, str(self.video) if actual == 'local' else self.remote
    def meta(self, mode):
        actual, _ = self.media(mode); result = dict(self.meta_base); result['source'] = actual; result['available_sources'] = dict(local=bool(self.video and self.video.is_file()), dropbox=bool(self.remote))
        if actual == 'local':
            if self.probe is None:
                data = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,avg_frame_rate:format=duration','-of','json',str(self.video)], timeout=25))
                stream = data['streams'][0]; a,b = map(float, stream['avg_frame_rate'].split('/'))
                if abs(a/b-self.fps) > .01: raise ValueError('Configured FPS does not match local video average FPS')
                self.probe = dict(width=stream['width'], height=stream['height'], duration=float(data['format']['duration']))
            result.update(self.probe)
        result['metadata_estimated'] = actual == 'dropbox'
        return result
    def score_chunk(self, spec, start, end):
        p = spec['path']; fc,sc = spec['frame'],spec['score']
        names = fields(p) if p.suffix == '.parquet' else list(pd.read_csv(p,nrows=0))
        vc = spec['video'] or next((c for c in ('VideoID','video_id','video','day_label') if c in names), None)
        if p.suffix == '.parquet':
            filters = [(fc, '>=', start), (fc, '<=', end)]
            if vc: filters.append((vc, '==', self.video_id))
            df = pd.read_parquet(p, columns=[fc,sc], filters=filters)
        else:
            df = pd.read_csv(p)
            if vc: df = df[df[vc].astype(str)==self.video_id]
            df = df[df[fc].between(start,end)]
        if df[fc].duplicated().any(): raise ValueError('Duplicate score frame')
        result = {}
        for f,s in zip(df[fc],df[sc]):
            if pd.isna(s): continue
            if not np.isfinite(s) or not 0 <= s <= 1 or not np.isfinite(f) or int(f)!=f or f<0: raise ValueError('Invalid score or frame')
            result[str(int(f))] = float(s)
        return result
    def chunk(self, index, layer='raw'):
        if index < 0 or index > 100000: raise ValueError('Invalid chunk index')
        if layer not in {'raw', 'cleaned'}: raise ValueError('Unknown pose layer')
        key = index, layer
        with self.lock:
            if key in self.chunks: self.chunks.move_to_end(key); return self.chunks[key]
            start = index*self.meta_base['chunk_frames']; end = start+self.meta_base['chunk_frames']-1
            frames = {}; warnings = []; scores = {}; features = []
            if layer in self.layers:
                p=self.layers[layer]; keys=self.keys[layer]
                df=scoped(p,start-1,end,self.video_id,self.cfg.get('expected_project_id'),['FrameNum','TrackUID','X_center','Y_center']+[k+s for k in keys for s in ('_x','_y')])
                if df.duplicated(['FrameNum','TrackUID']).any(): raise ValueError('Duplicate pose frame/TrackUID')
                for r in df.to_dict('records'):
                    frames.setdefault(str(int(r['FrameNum'])), []).append(dict(id=str(r['TrackUID']),center=[r['X_center'],r['Y_center']],points=[[r[k+'_x'],r[k+'_y']] for k in keys]))
            for spec in self.score_specs:
                try: scores[spec['index']]=self.score_chunk(spec,start,end)
                except Exception as exc: warnings.append(spec['name']+' score data rejected: '+str(exc))
            if self.feature_names:
                try:
                    df=scoped(self.feature_path,start,end,self.video_id,self.cfg.get('expected_project_id'),['FrameNum','TrackUID_1','TrackUID_2']+self.feature_names)
                    if len(df)>100000: raise ValueError('Too many pair rows in one section')
                    if df.duplicated(['FrameNum','TrackUID_1','TrackUID_2']).any(): raise ValueError('Duplicate pair feature keys')
                    for r in df.to_dict('records'):
                        features.append(dict(frame=int(r['FrameNum']),a=str(r['TrackUID_1']),b=str(r['TrackUID_2']),values={k:r[k] for k in self.feature_names}))
                except Exception as exc: warnings.append('Shared feature data rejected: '+str(exc))
            obj=dict(start=start,end=end,frames=frames,scores=scores,features=features,warnings=warnings)
            data=json_bytes(obj); self.chunks[key]=data
            while len(self.chunks)>6: self.chunks.popitem(last=False)
            return data

"""Recording selector, local/remote video playback, and synchronized project outputs.
Run from the repository root: python -m visualization.live
"""
import argparse
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import socket
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, parse_qs
from urllib.request import Request, urlopen
from .catalog import Catalog, json_bytes


def byte_range(header, size):
    if not header: return 0, size-1, False
    m = re.fullmatch(r'bytes=(\d*)-(\d*)', header)
    if not m or not any(m.groups()): raise ValueError('Invalid range')
    a,b = m.groups()
    if not a:
        if int(b)<=0: raise ValueError('Invalid suffix')
        return max(0,size-int(b)),size-1,True
    start=int(a); end=min(int(b) if b else size-1,size-1)
    if start>=size or end<start: raise ValueError('Unsatisfiable range')
    return start,end,True

class App:
    def __init__(self, config): self.config=Path(config).resolve(); self.catalog=Catalog(config)
    def refresh(self): self.catalog=Catalog(self.config)

class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    def __init__(self,*args,app,**kwargs): self.app=app; super().__init__(*args,**kwargs)
    def handle(self):
        try: super().handle()
        except (ConnectionResetError, BrokenPipeError): pass
    def log_message(self,*args): pass
    def do_GET(self): self.serve(False)
    def do_HEAD(self): self.serve(True)
    def reply(self,data,status=200,content_type='application/json',head=False):
        self.send_response(status); self.send_header('Content-Type',content_type); self.send_header('Content-Length',str(len(data))); self.send_header('Cache-Control','no-store'); self.end_headers()
        if not head: self.wfile.write(data)
    def local_video(self,path,head):
        size=path.stat().st_size
        try: start,end,part=byte_range(self.headers.get('Range'),size)
        except ValueError:
            self.send_response(416); self.send_header('Content-Range',f'bytes */{size}'); self.send_header('Content-Length','0'); self.end_headers(); return
        self.send_response(206 if part else 200); self.send_header('Content-Type',mimetypes.guess_type(str(path))[0] or 'video/mp4'); self.send_header('Accept-Ranges','bytes'); self.send_header('Content-Length',str(end-start+1))
        if part: self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
        self.end_headers()
        if head: return
        with path.open('rb') as f:
            f.seek(start); remaining=end-start+1
            while remaining:
                data=f.read(min(1024*1024,remaining))
                if not data: break
                self.wfile.write(data); remaining-=len(data)
    def remote_video(self,url,head):
        requested=self.headers.get('Range')
        if requested and not re.fullmatch(r'bytes=(?:\d+-\d*|-\d+)',requested): raise ValueError('Only a single byte range is supported')
        headers={'User-Agent':'CichlidVisualization/2','Accept-Encoding':'identity'}
        if requested: headers['Range']=requested
        try: upstream=urlopen(Request(url,headers=headers,method='HEAD' if head else 'GET'),timeout=25)
        except HTTPError as exc:
            status=416 if exc.code==416 else 502
            self.reply(json_bytes({'error':f'Remote video returned HTTP {exc.code}. Check the shared link and permissions.'}),status,head=head);return
        except (URLError,TimeoutError,socket.timeout):
            self.reply(json_bytes({'error':'Remote video connection failed. Retry or switch to Local.'}),502,head=head);return
        with upstream:
            status=upstream.status; typ=upstream.headers.get('Content-Type','application/octet-stream')
            if 'text/html' in typ.lower():
                self.reply(json_bytes({'error':'Remote link returned a web page instead of video. Use a direct shared-file video link.'}),502,head=head);return
            if status not in (200,206):
                self.reply(json_bytes({'error':'Unexpected response from remote video source'}),502,head=head);return
            if requested:
                cr=upstream.headers.get('Content-Range',''); m=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+|\*)',cr)
                if status!=206 or not m:
                    self.reply(json_bytes({'error':'Remote host did not honor seeking (HTTP byte ranges). Use a local copy.'}),502,head=head);return
                first=re.match(r'bytes=(\d+)-',requested)
                if first and int(first[1])!=int(m[1]):
                    self.reply(json_bytes({'error':'Remote host returned a different byte range; refusing misaligned playback.'}),502,head=head);return
            self.send_response(status); self.send_header('Content-Type',typ)
            for name in ('Content-Length','Content-Range','Accept-Ranges','ETag','Last-Modified'):
                if upstream.headers.get(name): self.send_header(name,upstream.headers[name])
            if not upstream.headers.get('Content-Length'):
                self.send_header('Connection','close'); self.close_connection=True
            self.end_headers()
            if head: return
            try:
                while True:
                    data=upstream.read(256*1024)
                    if not data: break
                    self.wfile.write(data)
            except (URLError,TimeoutError,socket.timeout): self.close_connection=True
    def serve(self,head):
        parsed=urlparse(self.path); q=parse_qs(parsed.query)
        try:
            if parsed.path=='/': return self.reply(Path(__file__).with_name('live.html').read_bytes(),content_type='text/html; charset=utf-8',head=head)
            if parsed.path=='/api/catalog':
                if q.get('refresh')==['1']: self.app.refresh()
                return self.reply(json_bytes(self.app.catalog.listing()),head=head)
            rec=self.app.catalog.get(q.get('recording',[None])[0])
            mode=q.get('source',['auto'])[0]
            if parsed.path=='/api/meta': return self.reply(json_bytes(rec.meta(mode)),head=head)
            if parsed.path=='/api/chunk': return self.reply(rec.chunk(int(q.get('index',['0'])[0]),q.get('layer',['raw'])[0]),head=head)
            if parsed.path=='/video':
                actual,url=rec.media(mode)
                return self.local_video(rec.video,head) if actual=='local' else self.remote_video(url,head)
            return self.reply(json_bytes({'error':'Not found'}),404,head=head)
        except (BrokenPipeError,ConnectionResetError): pass
        except Exception as exc:
            self.reply(json_bytes({'error':str(exc)}),400,head=head)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',nargs='?',default=str(Path(__file__).with_name('recordings.json')))
    parser.add_argument('--port',type=int,default=int(os.environ.get('PORT','8001')))
    parser.add_argument('--host',default=os.environ.get('HOST','127.0.0.1'))
    args=parser.parse_args()
    try: app=App(args.config)
    except Exception as exc: parser.exit(1,f'Catalog error: {exc}\n')
    server=ThreadingHTTPServer((args.host,args.port),partial(Handler,app=app)); server.daemon_threads=True
    print(f'Listening on {args.host}:{args.port}. No model training. Ctrl+C to stop.',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
if __name__=='__main__': main()

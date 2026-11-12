"""Download the source videos (and optionally pose tables) in parallel, e.g. on PACE.

Each file is split into byte ranges fetched concurrently, so one 33 GB video uses many
connections. Downloads resume: finished chunks are recorded next to the ``.part`` file.
Standard library only, so it runs before ``uv sync``.

    python download_data.py                      # videos in video_source.VIDEO_URLS
    python download_data.py --tables             # plus the parquet tables
    python download_data.py --urls extra.txt     # plus "<filename> <url>" lines
    python download_data.py --workers 32 --dest ~/scratch/cichlid-analysis/data
"""

import argparse
import ast
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
CHUNK = 256 * 1024 * 1024
BLOCK = 4 * 1024 * 1024
RETRIES = 6
HEADERS = {"User-Agent": "cichlid-analysis-downloader"}


def known_urls(include_tables):
    """Read the URL dicts from video_source.py without importing OpenCV."""
    tree = ast.parse((ROOT / "video_source.py").read_text())
    wanted = {"VIDEO_URLS"} | ({"TABLE_URLS"} if include_tables else set())
    urls = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) in wanted for t in node.targets):
            urls.update(ast.literal_eval(node.value))
    return urls


def read_url_file(path):
    urls = {}
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            name, url = line.split(maxsplit=1)
            urls[name] = url.strip()
    return urls


def remote_size(url):
    """Total size from a 1-byte range request; also checks the server honours ranges."""
    with urlopen(Request(url, headers={**HEADERS, "Range": "bytes=0-0"}), timeout=60) as r:
        if r.status != 206:
            raise RuntimeError(f"server ignored Range request (HTTP {r.status})")
        if "text/html" in r.headers.get("Content-Type", ""):
            raise RuntimeError("got an HTML page instead of a file; check the link")
        return int(r.headers["Content-Range"].rsplit("/", 1)[1])


class Download:
    def __init__(self, name, url, dest):
        self.name, self.url = name, url
        self.final = dest / name
        self.part = dest / (name + ".part")
        self.state = dest / (name + ".part.json")
        self.lock = threading.Lock()
        self.size = remote_size(url)
        self.done = set()
        if self.part.exists() and self.state.exists():
            saved = json.loads(self.state.read_text())
            if saved.get("size") == self.size and saved.get("chunk") == CHUNK:
                self.done = set(saved["done"])
        with open(self.part, "ab") as f:
            f.truncate(self.size)
        self.chunks = [i for i in range((self.size + CHUNK - 1) // CHUNK) if i not in self.done]

    def fetch(self, index, progress):
        start = index * CHUNK
        end = min(start + CHUNK, self.size) - 1
        for attempt in range(RETRIES):
            pos = start
            try:
                req = Request(self.url, headers={**HEADERS, "Range": f"bytes={start}-{end}"})
                fd = os.open(self.part, os.O_WRONLY)
                try:
                    with urlopen(req, timeout=120) as r:
                        if r.status != 206:
                            raise RuntimeError(f"HTTP {r.status} for a range request")
                        while block := r.read(BLOCK):
                            os.pwrite(fd, block, pos)
                            pos += len(block)
                            progress.add(len(block))
                finally:
                    os.close(fd)
                if pos != end + 1:
                    raise RuntimeError(f"short read ({pos - start} of {end - start + 1} bytes)")
                with self.lock:
                    self.done.add(index)
                    self.state.write_text(json.dumps({"size": self.size, "chunk": CHUNK, "done": sorted(self.done)}))
                return
            except Exception as exc:
                progress.add(start - pos)  # roll back partial bytes of the failed attempt
                if attempt == RETRIES - 1:
                    raise RuntimeError(f"{self.name} chunk {index}: {exc}") from exc
                time.sleep(2 ** attempt)

    def finish(self):
        if len(self.done) * CHUNK >= self.size:
            self.part.replace(self.final)
            self.state.unlink(missing_ok=True)
            return True
        return False


class Progress:
    def __init__(self, total):
        self.total, self.bytes, self.lock = total, 0, threading.Lock()
        self.start = self.last = time.monotonic()

    def add(self, n):
        with self.lock:
            self.bytes += n
            now = time.monotonic()
            if now - self.last >= 10:
                self.last = now
                rate = self.bytes / (now - self.start) / 1e6
                print(f"  {self.bytes / 1e9:.1f} / {self.total / 1e9:.1f} GB  ({rate:.0f} MB/s)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=ROOT / "data")
    parser.add_argument("--workers", type=int, default=16, help="concurrent range requests across all files")
    parser.add_argument("--tables", action="store_true", help="also download the parquet tables")
    parser.add_argument("--urls", type=Path, help='extra "<filename> <url>" lines (Dropbox links ending dl=1)')
    args = parser.parse_args()

    urls = known_urls(args.tables)
    if args.urls:
        urls.update(read_url_file(args.urls))
    args.dest.mkdir(parents=True, exist_ok=True)

    downloads = []
    for name, url in urls.items():
        if (args.dest / name).exists():
            print(f"skip {name} (already downloaded)")
            continue
        try:
            downloads.append(Download(name, url, args.dest))
        except Exception as exc:
            print(f"FAILED {name}: {exc}", file=sys.stderr)
    if not downloads:
        return

    remaining = sum(min(CHUNK, d.size - i * CHUNK) for d in downloads for i in d.chunks)
    print(f"Downloading {len(downloads)} files, {remaining / 1e9:.1f} GB left, {args.workers} workers")
    progress = Progress(remaining)
    failed = False
    with ThreadPoolExecutor(args.workers) as pool:
        jobs = {pool.submit(d.fetch, i, progress): d for d in downloads for i in d.chunks}
        for job in as_completed(jobs):
            try:
                job.result()
            except Exception as exc:
                failed = True
                print(f"FAILED {exc}", file=sys.stderr)
    for d in downloads:
        print(f"{'done' if d.finish() else 'INCOMPLETE (rerun to resume)'}  {d.name}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

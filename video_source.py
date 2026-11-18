"""Locate source videos and read frame ranges from local files or by HTTP range streaming."""

from pathlib import Path
from time import sleep
from urllib.request import urlretrieve

import cv2

from models.logistic_regression.train import VIDEO_FRAMES

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RETRIES = 3

# Source videos (~31 GB each) are streamed, so a clip fetches only about 1 MB per second of video.
VIDEO_URLS = {
    "0028_vid.mp4": "https://www.dropbox.com/scl/fo/zdkqd0n1tt6i9hf9wyta9/AMkLWMbEcHxMGEXGPuxG1NI/0028_vid.mp4?rlkey=l0gupg6wkgdd0fq5dbycuqm8v&st=m42jq5db&dl=1",
    "0031_vid.mp4": "https://www.dropbox.com/scl/fo/zdkqd0n1tt6i9hf9wyta9/ADCA61tJU9SWYSnh9DNWkkk/0031_vid.mp4?rlkey=l0gupg6wkgdd0fq5dbycuqm8v&st=xmrzc3hs&dl=1",
}
TABLE_URLS = {
    "0028_vid.parquet": "https://www.dropbox.com/scl/fo/zdkqd0n1tt6i9hf9wyta9/AIzGAhjimK-vvUN1IRetNrI/0028_vid.parquet?rlkey=l0gupg6wkgdd0fq5dbycuqm8v&st=zd96dl6g&dl=1",
    "0028_vid_pairs.parquet": "https://www.dropbox.com/scl/fo/zdkqd0n1tt6i9hf9wyta9/AL6msxoSPODIp3jfuXxWb6Q/0028_vid_pairs.parquet?rlkey=l0gupg6wkgdd0fq5dbycuqm8v&st=nb7wminh&dl=1",
    "0031_vid_pairs.parquet": "https://www.dropbox.com/scl/fo/zdkqd0n1tt6i9hf9wyta9/AOLNU5Q-ydFtOxPr3Bppvss/0031_vid_pairs.parquet?rlkey=l0gupg6wkgdd0fq5dbycuqm8v&st=pkqm83so&dl=1",
    "0031_vid.parquet": "https://www.dropbox.com/scl/fo/zdkqd0n1tt6i9hf9wyta9/ACNtNV3xM1k06lXgc4zpB0o/0031_vid.parquet?rlkey=l0gupg6wkgdd0fq5dbycuqm8v&st=nox9c3q0&dl=1",
}


def download(url, destination):
    """Download to a .part file first so interrupted downloads are retried."""
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {destination.name}...")
    try:
        urlretrieve(url, partial)
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return destination


def resolve_video(path):
    """Return the local video if present, else its streaming URL."""
    path = Path(path)
    if path.exists():
        return str(path)
    if path.name in VIDEO_URLS:
        return VIDEO_URLS[path.name]
    raise FileNotFoundError(f"Video not found and has no streaming URL: {path}")


def video_frame_count(path):
    """Frame count, avoiding opening the video (~13 s when streaming) if it is known."""
    path = Path(path)
    if path.stem in VIDEO_FRAMES:
        return VIDEO_FRAMES[path.stem]
    with VideoReader(path) as reader:
        return reader.frame_count


class VideoReader:
    """OpenCV capture for a local or streamed video that resumes dropped streams."""

    def __init__(self, path, retries=RETRIES):
        self.name = Path(path).name
        self.source = resolve_video(path)
        self.retries = retries if self.source.startswith("http") else 0
        self.capture = self._open()
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()

    def release(self):
        self.capture.release()

    def _open(self):
        capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open {self.name}")
        return capture

    def _reopen(self, frame_number):
        self.capture.release()
        try:
            self.capture = self._open()
        except RuntimeError:
            return False
        return self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    def frames(self, first, last):
        """Yield (frame_number, BGR frame) for an inclusive range, reopening failed streams."""
        positioned = self.capture.set(cv2.CAP_PROP_POS_FRAMES, first)
        failures = 0
        frame_number = first
        while frame_number <= last:
            # Never read after a failed seek: the frame would be misnumbered.
            ok, frame = self.capture.read() if positioned else (False, None)
            if ok:
                yield frame_number, frame
                frame_number += 1
                failures = 0
                continue
            failures += 1
            if failures > self.retries:
                raise RuntimeError(f"Could not decode frame {frame_number} from {self.name}")
            sleep(2 ** (failures - 1))
            positioned = self._reopen(frame_number)

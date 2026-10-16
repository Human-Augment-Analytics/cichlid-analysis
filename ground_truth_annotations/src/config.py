"""Project-wide constants and paths."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = PROJECT_ROOT / "images"
CLIPS_DIR = DATA_DIR / "circling-events"

FPS = 30.0
SEED = 42

PROJECT_TO_VIDEO = {
    "MC920": "0028_vid",
    "F1_613": "0031_vid",
}

KP_NAMES = ["Nose", "LeftEye", "RightEye", "Head",
            "Spine1", "Spine2", "Spine3", "Spine4", "Peduncle", "TailTip"]

KP_COLOR = {
    "Nose": (0, 255, 255),
    "LeftEye": (255, 0, 255),
    "RightEye": (255, 255, 0),
    "Head": (0, 140, 255),
    "Spine1": (0, 255, 0),
    "Spine2": (255, 200, 0),
    "Spine3": (128, 255, 128),
    "Spine4": (60, 100, 255),
    "Peduncle": (200, 120, 255),
    "TailTip": (255, 255, 255),
}

SKELETON = [
    ("Nose", "Head"), ("Head", "LeftEye"), ("Head", "RightEye"),
    ("Head", "Spine1"), ("Spine1", "Spine2"), ("Spine2", "Spine3"),
    ("Spine3", "Spine4"), ("Spine4", "Peduncle"), ("Peduncle", "TailTip"),
]

COLOR_MALE = (255, 120, 60)
COLOR_FEMALE = (80, 80, 255)
COLOR_OTHER = (200, 200, 200)

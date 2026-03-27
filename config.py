from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

# helpers
def parse_tuple(value, cast=float):
    return tuple(cast(x) for x in value.split(","))

def parse_list(value):
    return value.split(",")

# MODEL_NAME = os.getenv("MODEL_NAME", "default_name")

TARGET_SPACING = parse_tuple(os.getenv("TARGET_SPACING"), float)
TARGET_SHAPE = parse_tuple(os.getenv("TARGET_SHAPE"), int)

DATASET = Path(os.getenv("DATASET"))

OUT_IMG_DIR = DATASET / os.getenv("OUT_IMG_DIR")
OUT_MASK_DIR = DATASET / os.getenv("OUT_MASK_DIR")

META_PATH_UNPROCESSED = os.getenv("META_PATH_UNPROCESSED")
META_PATH = DATASET / os.getenv("META_PATH")
CHECKPOINT_DIR = DATASET / os.getenv("CHECKPOINT_DIR")

DATASET_LIST = parse_list(os.getenv("DATASET_LIST"))

EPOCH_NUMBER=int(os.getenv("EPOCH_NUMBER"))
EARLY_STOPPING_EPOCH=int(os.getenv("EARLY_STOPPING_EPOCH"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE"))

VALIDATE_SAMPLE= DATASET / os.getenv("VALIDATE_SAMPLE")

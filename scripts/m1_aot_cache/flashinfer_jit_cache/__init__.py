"""Private opt-in provider for the pinned FlashInfer AOT discovery API."""
import os
from pathlib import Path
from m1_private_aot import validate_cache

__version__ = "0.6.18.post1"


def get_jit_cache_dir():
    root = Path(os.environ["MEGARTX_M1_PRIVATE_AOT"])
    validate_cache(root)
    return str(root.resolve() / "aot")

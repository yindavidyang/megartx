"""Check the actual pinned vtable/PLT bindings after Torch, without a model."""
import ctypes
import json
import os
from pathlib import Path
import sys

import torch

library, module, output = map(Path, sys.argv[1:])
if torch.__version__ != "2.13.0+cu130":
    raise RuntimeError("installed Torch runtime pin changed")
if os.environ.get("LD_PRELOAD"):
    raise RuntimeError("binding control requires ordinary Torch initialization")
os.environ["MEGARTX_M1_STOCK_MODULE"] = str(module.resolve())
native = ctypes.CDLL(str(library.resolve()), mode=ctypes.RTLD_GLOBAL)
native.megartx_m1_verify_bindings.argtypes = [ctypes.c_char_p]
native.megartx_m1_error.restype = ctypes.c_char_p
status = native.megartx_m1_verify_bindings(os.fsencode(library.resolve()))
if status:
    raise RuntimeError(native.megartx_m1_error().decode())
record = {"scope": "compiled pinned native relocations; no model or CUDA submission",
          "torch_runtime": torch.__version__, "loader": "after_torch_rtld_global",
          "relocations_bound_to_bridge": 4, "active_after": native.megartx_m1_active()}
if record["active_after"]:
    raise RuntimeError("binding control left an active lease")
with output.open("x") as stream:
    json.dump(record, stream, indent=2)
print(json.dumps(record), flush=True)

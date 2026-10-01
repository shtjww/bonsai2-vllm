"""List all non-v.blk (global) tensors of the mmproj GGUF."""
import sys

sys.path.insert(0, ".")
from gguf_reader import GGUFReader

with GGUFReader(sys.argv[1]) as r:
    for n in sorted(r.tensors.keys()):
        if not n.startswith("v.blk."):
            t = r.tensors[n]
            print(f"  {t.type_name:6s} {str(t.shape):24s} {n}")

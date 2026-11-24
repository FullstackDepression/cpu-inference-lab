"""Where the bytes of a GGUF model live: tensor groups by size.

    python3 bench/tensors.py model.gguf   (needs gguf-py from llama.cpp + numpy)
"""
import sys
from collections import defaultdict
from gguf import GGUFReader

groups = defaultdict(int)
for t in GGUFReader(sys.argv[1]).tensors:
    # blk.17.ffn_down.weight -> blk.*.ffn_down.weight: all layers in one row
    name = t.name if not t.name.startswith("blk.") else "blk.*." + t.name.split(".", 2)[2]
    groups[name] += int(t.n_bytes)

total = sum(groups.values())
for name, size in sorted(groups.items(), key=lambda g: -g[1])[:8]:
    print(f"{size / 2**30:6.2f} GiB  {size / total:4.0%}  {name}")
print(f"{total / 2**30:6.2f} GiB        total")

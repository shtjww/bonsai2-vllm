"""Minimal GGUF v3 reader (mmap-based, zero-copy for tensor data).

Supports the PrismML fork extensions: tensor types PQ2_0 (142) and PTQ1_0 (143),
plus prism.hadamard.* metadata keys.

Only implements what the Bonsai 2 project needs — not a general GGUF library.
"""
from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

GGUF_MAGIC = b"GGUF"

# ggml type enum -> (block_size_elements, block_size_bytes)
# only types we expect in Bonsai 2 checkpoints
GGML_TYPES = {
    0:  ("F32",   1,   4),
    1:  ("F16",   1,   2),
    30: ("BF16",  1,   2),
    8:  ("Q8_0",  32,  34),
    142: ("PQ2_0", 128, 34),   # Prism 2-bit container, group 128
    143: ("PTQ1_0", 128, 28),  # Prism ternary, group 128
}

# gguf metadata value types
VAL_U8, VAL_I8, VAL_U16, VAL_I16, VAL_U32, VAL_I32 = 0, 1, 2, 3, 4, 5
VAL_F32, VAL_BOOL, VAL_STRING, VAL_ARRAY, VAL_U64, VAL_I64, VAL_F64 = 6, 7, 8, 9, 10, 11, 12

_SCALAR_FMT = {
    VAL_U8: "<B", VAL_I8: "<b", VAL_U16: "<H", VAL_I16: "<h",
    VAL_U32: "<I", VAL_I32: "<i", VAL_U64: "<Q", VAL_I64: "<q",
    VAL_F32: "<f", VAL_F64: "<d",
}


@dataclass
class TensorInfo:
    name: str
    shape: tuple[int, ...]      # ggml order: ne[0] is fastest-varying
    ggml_type: int
    offset: int                  # relative to data start
    type_name: str = field(init=False)
    n_elements: int = field(init=False)
    n_bytes: int = field(init=False)

    def __post_init__(self):
        name, bs, bb = GGML_TYPES[self.ggml_type]
        self.type_name = name
        n = 1
        for d in self.shape:
            n *= d
        self.n_elements = n
        assert n % bs == 0, f"{self.name}: {n} not divisible by block {bs}"
        self.n_bytes = (n // bs) * bb


class GGUFReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._f = open(self.path, "rb")
        self.mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        self.metadata: dict = {}
        self.tensors: dict[str, TensorInfo] = {}
        self.data_start: int = 0
        self._parse()

    def close(self):
        self.mm.close()
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ---- parsing ----

    def _read(self, off: int, fmt: str):
        size = struct.calcsize(fmt)
        return struct.unpack_from(fmt, self.mm, off)[0], off + size

    def _read_string(self, off: int):
        n, off = self._read(off, "<Q")
        s = bytes(self.mm[off:off + n]).decode("utf-8")
        return s, off + n

    def _read_value(self, off: int, vtype: int):
        if vtype == VAL_BOOL:
            v, off = self._read(off, "<B")
            return bool(v), off
        if vtype in _SCALAR_FMT:
            return self._read(off, _SCALAR_FMT[vtype])
        if vtype == VAL_STRING:
            return self._read_string(off)
        if vtype == VAL_ARRAY:
            etype, off = self._read(off, "<I")
            n, off = self._read(off, "<Q")
            out = []
            for _ in range(n):
                v, off = self._read_value(off, etype)
                out.append(v)
            return out, off
        raise ValueError(f"unknown gguf value type {vtype}")

    def _parse(self):
        off = 0
        magic = bytes(self.mm[0:4])
        assert magic == GGUF_MAGIC, f"not a GGUF file: {self.path}"
        off = 4
        self.version, off = self._read(off, "<I")
        assert self.version == 3, f"unsupported GGUF version {self.version}"
        n_tensors, off = self._read(off, "<Q")
        n_kv, off = self._read(off, "<Q")

        for _ in range(n_kv):
            key, off = self._read_string(off)
            vtype, off = self._read(off, "<I")
            self.metadata[key], off = self._read_value(off, vtype)

        for _ in range(n_tensors):
            name, off = self._read_string(off)
            n_dims, off = self._read(off, "<I")
            dims = []
            for _ in range(n_dims):
                d, off = self._read(off, "<Q")
                dims.append(d)
            gtype, off = self._read(off, "<I")
            toff, off = self._read(off, "<Q")
            if gtype not in GGML_TYPES:
                raise ValueError(f"tensor {name}: unsupported ggml type {gtype} (add to GGML_TYPES)")
            self.tensors[name] = TensorInfo(name, tuple(dims), gtype, toff)

        # data section starts at next ALIGNMENT(32)-byte boundary
        align = self.metadata.get("general.alignment", 32)
        self.data_start = (off + align - 1) // align * align

    # ---- tensor access ----

    def tensor_bytes(self, name: str) -> memoryview:
        t = self.tensors[name]
        beg = self.data_start + t.offset
        return self.mm[beg:beg + t.n_bytes]

    def tensor_f16(self, name: str) -> np.ndarray:
        """F16/BF16/F32 tensor -> float32 numpy array (ggml dim order).

        WARNING: for ndim>1 the C-order reshape to ggml shape SCRAMBLES data
        (ggml is ne0-fastest). Use tensor_torch() for HF/torch orientation.
        """
        return self._tensor_flat_f32(name).reshape(self.tensors[name].shape)

    def tensor_torch(self, name: str) -> np.ndarray:
        """F16/BF16/F32 tensor -> float32 numpy array with axes REVERSED
        (torch/HF convention: 2D linear weights come out as (out, in))."""
        return self._tensor_flat_f32(name).reshape(self.tensors[name].shape[::-1])

    def _tensor_flat_f32(self, name: str) -> np.ndarray:
        t = self.tensors[name]
        buf = self.tensor_bytes(name)
        if t.ggml_type == 1:   # F16
            arr = np.frombuffer(buf, dtype=np.float16).astype(np.float32)
        elif t.ggml_type == 30:  # BF16
            u16 = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32)
            arr = (u16 << 16).view(np.float32)
        elif t.ggml_type == 0:  # F32
            arr = np.frombuffer(buf, dtype=np.float32)
        else:
            raise ValueError(f"{name} is {t.type_name}, not a float type")
        return arr

    # ---- prism metadata helpers ----

    @property
    def hadamard_config(self) -> dict | None:
        if "prism.hadamard.version" not in self.metadata:
            return None
        keys = [k for k in self.metadata if k.startswith("prism.")]
        return {k: self.metadata[k] for k in sorted(keys)}

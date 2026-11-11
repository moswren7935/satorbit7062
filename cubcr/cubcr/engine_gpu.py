from __future__ import annotations

import ctypes as ct
import os

import numpy as np

from .engine_cpu import CModel, model_struct, prepare_masks, _ensure_built, CSRC
from .model import CompiledCSG


class GPUStats(ct.Structure):
    _fields_ = [("total_ms", ct.c_double), ("kernel_ms", ct.c_double), ("batch", ct.c_int64),
                ("waves", ct.c_int), ("rmax_class", ct.c_int), ("tpb", ct.c_int),
                ("bytes_per_wave", ct.c_size_t), ("l2_tile", ct.c_int64), ("table_bytes", ct.c_size_t),
                ("graph_captures", ct.c_int)]

    def as_dict(self):
        return {f: getattr(self, f) for f, _ in self._fields_}


_lib = None


def lib():
    global _lib
    if _lib is None:
        _lib = ct.CDLL(os.environ.get("CUBCR_GPU_LIB") or _ensure_built("gpu"))
        _lib.cubcr_gpu_create.restype = ct.c_void_p
        _lib.cubcr_gpu_create.argtypes = [ct.POINTER(CModel), ct.c_int]
        _lib.cubcr_gpu_destroy.argtypes = [ct.c_void_p]
        _lib.cubcr_gpu_solve.restype = ct.c_int
        _lib.cubcr_gpu_solve.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_int, ct.c_int64,
                                         ct.c_void_p, ct.c_int, ct.c_int64, ct.c_int, ct.c_int, ct.c_int64]
        _lib.cubcr_gpu_last_stats.argtypes = [ct.c_void_p, ct.POINTER(GPUStats)]
        _lib.cubcr_gpu_last_error.restype = ct.c_char_p
        _lib.cubcr_gpu_mem_info.argtypes = [ct.c_int, ct.POINTER(ct.c_size_t), ct.POINTER(ct.c_size_t)]
        _lib.cubcr_gpu_device_name.argtypes = [ct.c_int, ct.c_char_p, ct.c_int]
        _lib.cubcr_gpu_device_count.restype = ct.c_int
    return _lib


def device_count() -> int:
    return lib().cubcr_gpu_device_count()


def device_name(device=0) -> str:
    buf = ct.create_string_buffer(256)
    lib().cubcr_gpu_device_name(device, buf, 256)
    return buf.value.decode()


def mem_info(device=0):
    f, t = ct.c_size_t(), ct.c_size_t()
    rc = lib().cubcr_gpu_mem_info(device, ct.byref(f), ct.byref(t))
    if rc:
        raise RuntimeError(lib().cubcr_gpu_last_error().decode())
    return f.value, t.value


class GPUEngine:
    def __init__(self, compiled: CompiledCSG, device: int = 0, dtype: str = "fp64",
                 max_batch: int = 0, use_graph: bool = False, tpb: int = 0, l2_tile: int = 0):


        if l2_tile and l2_tile > 0 and use_graph:


            raise ValueError("l2_tile > 0 with use_graph=True is not supported (nondeterministic)")
        self.compiled = compiled
        self.device = device
        self.dtype = dtype
        self.max_batch = max_batch
        self.use_graph = use_graph
        self.tpb = tpb
        self.l2_tile = l2_tile
        m, keep = model_struct(compiled)
        self._ctx = lib().cubcr_gpu_create(ct.byref(m), device)
        if not self._ctx:
            raise RuntimeError("cubcr_gpu_create failed: " + lib().cubcr_gpu_last_error().decode())
        self.stats = None

    def close(self):
        if getattr(self, "_ctx", None):
            lib().cubcr_gpu_destroy(self._ctx)
            self._ctx = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def solve(self, masks, k: int | None = None) -> np.ndarray:
        if k is None:
            k = self.compiled.horizon_k
        arr, stride, B = prepare_masks(masks, k)
        out = np.empty(B, dtype=np.float64)
        rc = lib().cubcr_gpu_solve(self._ctx, int(k), arr.ctypes.data, int(stride), int(B),
                                   out.ctypes.data, 0 if self.dtype == "fp64" else 1,
                                   int(self.max_batch), 1 if self.use_graph else 0, int(self.tpb),
                                   int(self.l2_tile))
        if rc != 0:
            raise RuntimeError(f"cubcr_gpu_solve failed rc={rc}: {lib().cubcr_gpu_last_error().decode()}")
        st = GPUStats()
        lib().cubcr_gpu_last_stats(self._ctx, ct.byref(st))
        self.stats = st.as_dict()
        return out

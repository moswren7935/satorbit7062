from __future__ import annotations

import ctypes as ct
import os
import subprocess

import numpy as np

from .model import CompiledCSG, DMAX, AMAX

_HERE = os.path.dirname(os.path.abspath(__file__))
CSRC = os.path.join(os.path.dirname(_HERE), "csrc")
LIB_PATH = os.path.join(CSRC, "build", "libcubcr_cpu.so")


class CModel(ct.Structure):
    _fields_ = [("n_agents", ct.c_int32), ("n_states", ct.c_int64), ("init_state", ct.c_int64),
                ("n_rows", ct.c_int64), ("nnz", ct.c_int64),
                ("state_kind", ct.c_void_p), ("state_row_start", ct.c_void_p),
                ("state_agent", ct.c_void_p), ("state_radix", ct.c_void_p),
                ("weights", ct.c_void_p), ("row_ptr", ct.c_void_p), ("col", ct.c_void_p),
                ("val", ct.c_void_p)]


def _ensure_built(target="cpu"):
    path = LIB_PATH if target == "cpu" else os.path.join(CSRC, "build", "libcubcr_gpu.so")
    src = os.path.join(CSRC, "cubcr_cpu.cpp" if target == "cpu" else "cubcr_gpu.cu")
    if not os.path.exists(path) or os.path.getmtime(path) < max(os.path.getmtime(src),
                                                               os.path.getmtime(os.path.join(CSRC, "cubcr.h"))):
        subprocess.run(["make", "-C", CSRC, target], check=True)
    return path


def model_struct(c: CompiledCSG):

    keep = [np.ascontiguousarray(c.state_kind, dtype=np.uint8),
            np.ascontiguousarray(c.state_row_start, dtype=np.int64),
            np.ascontiguousarray(c.state_agent, dtype=np.int16),
            np.ascontiguousarray(c.state_radix, dtype=np.uint8),
            np.ascontiguousarray(c.weights, dtype=np.float64),
            np.ascontiguousarray(c.row_ptr, dtype=np.int64),
            np.ascontiguousarray(c.col, dtype=np.int32),
            np.ascontiguousarray(c.val, dtype=np.float64)]
    assert keep[2].shape == (c.n_states, DMAX) and keep[4].shape == (c.n_states, DMAX, AMAX)
    m = CModel(n_agents=c.n_agents, n_states=c.n_states, init_state=c.init_state,
               n_rows=c.n_rows, nnz=c.nnz,
               state_kind=keep[0].ctypes.data, state_row_start=keep[1].ctypes.data,
               state_agent=keep[2].ctypes.data, state_radix=keep[3].ctypes.data,
               weights=keep[4].ctypes.data, row_ptr=keep[5].ctypes.data,
               col=keep[6].ctypes.data, val=keep[7].ctypes.data)
    return m, keep


def prepare_masks(masks, k: int):


    if isinstance(masks, np.ndarray) and masks.ndim == 2:
        if masks.shape[0] != k:
            raise ValueError(f"step-level masks must have shape (k={k}, B)")
        return np.ascontiguousarray(masks, dtype=np.uint64), 1, masks.shape[1]
    if isinstance(masks, np.ndarray) and masks.ndim == 1:
        return np.ascontiguousarray(masks, dtype=np.uint64), 0, masks.shape[0]
    masks = list(masks)
    if all(isinstance(m, (int, np.integer)) for m in masks):
        return np.asarray(masks, dtype=np.uint64), 0, len(masks)
    arr = np.stack([np.asarray(m, dtype=np.uint64) for m in masks], axis=1)
    if arr.shape[0] != k:
        raise ValueError("each step-level mask must have k columns")
    return np.ascontiguousarray(arr), 1, arr.shape[1]


class CPUEngine:


    def __init__(self, compiled: CompiledCSG, n_threads: int = 0, block: int = 8, dtype="fp64"):
        self.lib = ct.CDLL(_ensure_built("cpu"))
        L = self.lib
        L.cubcr_cpu_create.restype = ct.c_void_p
        L.cubcr_cpu_create.argtypes = [ct.POINTER(CModel)]
        L.cubcr_cpu_destroy.argtypes = [ct.c_void_p]
        L.cubcr_cpu_solve_ctx.restype = ct.c_int
        L.cubcr_cpu_solve_ctx.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_int, ct.c_int64,
                                          ct.c_void_p, ct.c_int, ct.c_int, ct.c_int]
        L.cubcr_cpu_plan_bytes.restype = ct.c_int64
        L.cubcr_cpu_plan_bytes.argtypes = [ct.c_void_p]
        L.cubcr_cpu_last_error.restype = ct.c_char_p
        L.cubcr_cpu_max_threads.restype = ct.c_int
        self.compiled = compiled
        self.m, self._keep = model_struct(compiled)
        self.n_threads = n_threads
        self.block = block
        self.dtype = dtype
        self.max_threads = L.cubcr_cpu_max_threads()
        self._ctx = L.cubcr_cpu_create(ct.byref(self.m))
        if not self._ctx:
            raise RuntimeError("cubcr_cpu_create failed: " + L.cubcr_cpu_last_error().decode())
        self.plan_bytes = L.cubcr_cpu_plan_bytes(self._ctx)

    def close(self):
        if getattr(self, "_ctx", None):
            self.lib.cubcr_cpu_destroy(self._ctx)
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
        rc = self.lib.cubcr_cpu_solve_ctx(self._ctx, int(k), arr.ctypes.data, int(stride), int(B), out.ctypes.data,
                                          0 if self.dtype == "fp64" else 1, int(self.n_threads), int(self.block))
        if rc != 0:
            raise RuntimeError(f"cubcr_cpu_solve failed rc={rc}: {self.lib.cubcr_cpu_last_error().decode()}")
        return out

from __future__ import annotations

import ctypes as ct
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
E5_DIR = REPO / "rebuttal" / "engine" / "e5"
E5_LIB = E5_DIR / "build" / "libcubcr_e5_gpu.so"
D4 = 4
AMAX = 8
MODES = {"smem": 0, "acc": 1}


class CModel(ct.Structure):
    _fields_ = [("n_agents", ct.c_int32), ("n_states", ct.c_int64), ("init_state", ct.c_int64),
                ("n_rows", ct.c_int64), ("nnz", ct.c_int64),
                ("state_kind", ct.c_void_p), ("state_row_start", ct.c_void_p),
                ("state_agent", ct.c_void_p), ("state_radix", ct.c_void_p),
                ("weights", ct.c_void_p), ("row_ptr", ct.c_void_p), ("col", ct.c_void_p),
                ("val", ct.c_void_p)]


class GPUStats(ct.Structure):
    _fields_ = [("total_ms", ct.c_double), ("kernel_ms", ct.c_double), ("batch", ct.c_int64),
                ("waves", ct.c_int), ("rmax_class", ct.c_int), ("tpb", ct.c_int),
                ("bytes_per_wave", ct.c_size_t), ("l2_tile", ct.c_int64), ("table_bytes", ct.c_size_t),
                ("graph_captures", ct.c_int)]

    def as_dict(self):
        return {f: getattr(self, f) for f, _ in self._fields_}


class ClassInfo(ct.Structure):
    _fields_ = [("rmax_class", ct.c_int), ("tpb", ct.c_int), ("shm_bytes", ct.c_int64), ("n_states", ct.c_int64),
                ("fmax", ct.c_int), ("blocks_per_sm", ct.c_int), ("num_regs", ct.c_int),
                ("local_bytes", ct.c_int64), ("max_threads_per_block", ct.c_int)]

    def as_dict(self):
        return {f: getattr(self, f) for f, _ in self._fields_}


_lib = None


def lib():
    global _lib
    if _lib is None:
        if not E5_LIB.exists():
            raise FileNotFoundError(f"{E5_LIB} not built (make -C rebuttal/engine/e5)")
        L = ct.CDLL(str(E5_LIB))
        L.cubcr_e5_create.restype = ct.c_void_p
        L.cubcr_e5_create.argtypes = [ct.POINTER(CModel), ct.c_int, ct.c_int]
        L.cubcr_e5_destroy.argtypes = [ct.c_void_p]
        L.cubcr_e5_solve.restype = ct.c_int
        L.cubcr_e5_solve.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_int, ct.c_int64, ct.c_void_p,
                                     ct.c_int, ct.c_int64, ct.c_int, ct.c_int, ct.c_int64]
        L.cubcr_e5_last_stats.argtypes = [ct.c_void_p, ct.POINTER(GPUStats)]
        L.cubcr_e5_class_info_get.restype = ct.c_int
        L.cubcr_e5_class_info_get.argtypes = [ct.c_void_p, ct.c_int, ct.POINTER(ClassInfo)]
        L.cubcr_e5_mem_info.argtypes = [ct.c_int, ct.POINTER(ct.c_size_t), ct.POINTER(ct.c_size_t)]
        L.cubcr_e5_device_attr.argtypes = [ct.c_int] + [ct.POINTER(ct.c_int)] * 5
        L.cubcr_e5_last_error.restype = ct.c_char_p
        L.cubcr_e5_dmax.restype = ct.c_int
        L.cubcr_e5_rmax.restype = ct.c_int
        if L.cubcr_e5_dmax() != D4:
            raise RuntimeError("E5 library DMAX mismatch")
        _lib = L
    return _lib


def lib_sha256() -> str:
    return hashlib.sha256(open(E5_LIB, "rb").read()).hexdigest()


def mem_info(device: int = 0):
    f, t = ct.c_size_t(), ct.c_size_t()
    if lib().cubcr_e5_mem_info(device, ct.byref(f), ct.byref(t)):
        raise RuntimeError(lib().cubcr_e5_last_error().decode())
    return int(f.value), int(t.value)


def device_attr(device: int = 0) -> dict:
    vals = [ct.c_int() for _ in range(5)]
    if lib().cubcr_e5_device_attr(device, *[ct.byref(v) for v in vals]):
        raise RuntimeError(lib().cubcr_e5_last_error().decode())
    keys = ("smem_optin_per_block", "smem_per_sm", "regs_per_sm", "max_threads_per_sm", "sm_count")
    return {k: int(v.value) for k, v in zip(keys, vals)}


@dataclass
class Model4:

    n_agents: int
    n_states: int
    init_state: int
    horizon_k: int
    state_kind: np.ndarray
    state_row_start: np.ndarray
    state_agent: np.ndarray
    state_radix: np.ndarray
    weights: np.ndarray
    row_ptr: np.ndarray
    col: np.ndarray
    val: np.ndarray

    @property
    def n_rows(self):
        return int(self.row_ptr.shape[0] - 1)

    @property
    def nnz(self):
        return int(self.col.shape[0])

    @property
    def max_rows_per_state(self):
        return int(np.max(np.diff(self.state_row_start)))


def to_model4(c) -> Model4:

    S = int(c.n_states)
    ag = np.asarray(c.state_agent, dtype=np.int16)
    rx = np.asarray(c.state_radix, dtype=np.uint8)
    w = np.asarray(c.weights, dtype=np.float64)
    nsl = ag.shape[1]
    if nsl > D4:
        raise ValueError("more than 4 slots")
    if nsl < D4:
        ag = np.concatenate([ag, np.full((S, D4 - nsl), -1, dtype=np.int16)], axis=1)
        rx = np.concatenate([rx, np.ones((S, D4 - nsl), dtype=np.uint8)], axis=1)
        pad = np.zeros((S, D4 - nsl, AMAX), dtype=np.float64)
        pad[:, :, 0] = 1.0
        w = np.concatenate([w, pad], axis=1)
    return Model4(n_agents=int(c.n_agents), n_states=S, init_state=int(c.init_state), horizon_k=int(c.horizon_k),
                  state_kind=np.ascontiguousarray(c.state_kind, dtype=np.uint8),
                  state_row_start=np.ascontiguousarray(c.state_row_start, dtype=np.int64),
                  state_agent=np.ascontiguousarray(ag), state_radix=np.ascontiguousarray(rx),
                  weights=np.ascontiguousarray(w), row_ptr=np.ascontiguousarray(c.row_ptr, dtype=np.int64),
                  col=np.ascontiguousarray(c.col, dtype=np.int32), val=np.ascontiguousarray(c.val, dtype=np.float64))


def model_struct4(c: Model4):
    keep = [c.state_kind, c.state_row_start, c.state_agent, c.state_radix, c.weights, c.row_ptr, c.col, c.val]
    assert c.state_agent.shape == (c.n_states, D4) and c.weights.shape == (c.n_states, D4, AMAX)
    m = CModel(n_agents=c.n_agents, n_states=c.n_states, init_state=c.init_state, n_rows=c.n_rows, nnz=c.nnz,
               state_kind=keep[0].ctypes.data, state_row_start=keep[1].ctypes.data,
               state_agent=keep[2].ctypes.data, state_radix=keep[3].ctypes.data,
               weights=keep[4].ctypes.data, row_ptr=keep[5].ctypes.data, col=keep[6].ctypes.data,
               val=keep[7].ctypes.data)
    return m, keep


def _prepare_masks(masks, k):
    from cubcr.engine_cpu import prepare_masks
    return prepare_masks(masks, k)


class E5Engine:


    def __init__(self, model, mode: str = "acc", device: int = 0, dtype: str = "fp64", max_batch: int = 0,
                 use_graph: bool = True, tpb: int = 0):
        if dtype != "fp64":
            raise ValueError("E5 measurements are FP64 only")
        self.model4 = to_model4(model)
        self.compiled = model
        self.mode = mode
        self.device = device
        self.dtype = dtype
        self.max_batch = max_batch
        self.use_graph = use_graph
        self.tpb = tpb
        self._m, self._keep = model_struct4(self.model4)
        self._ctx = lib().cubcr_e5_create(ct.byref(self._m), device, MODES[mode])
        if not self._ctx:
            raise RuntimeError("cubcr_e5_create failed: " + lib().cubcr_e5_last_error().decode())
        self.stats = None

    def close(self):
        if getattr(self, "_ctx", None):
            lib().cubcr_e5_destroy(self._ctx)
            self._ctx = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def solve(self, masks, k=None) -> np.ndarray:
        if k is None:
            k = self.model4.horizon_k
        arr, stride, B = _prepare_masks(masks, k)
        out = np.empty(B, dtype=np.float64)
        rc = lib().cubcr_e5_solve(self._ctx, int(k), arr.ctypes.data, int(stride), int(B), out.ctypes.data, 0,
                                  int(self.max_batch), 1 if self.use_graph else 0, int(self.tpb), 0)
        if rc != 0:
            raise RuntimeError(f"cubcr_e5_solve failed rc={rc}: {lib().cubcr_e5_last_error().decode()}")
        st = GPUStats()
        lib().cubcr_e5_last_stats(self._ctx, ct.byref(st))
        self.stats = st.as_dict()
        return out

    def class_info(self):
        buf = (ClassInfo * 16)()
        n = lib().cubcr_e5_class_info_get(self._ctx, 16, buf)
        return [buf[i].as_dict() for i in range(max(0, n))]


def forward_error_bound_d(model, k=None, dtype: str = "fp64", dmax: int = D4):


    from cubcr.fp_error import FPErrorBound, UNIT_ROUNDOFF, gamma, lipschitz_rho
    if k is None:
        k = int(model.horizon_k)
    u = UNIT_ROUNDOFF[dtype]
    nnz_row = np.diff(model.row_ptr)
    max_nnz = int(nnz_row.max()) if nnz_row.size else 0
    max_rows = int(model.max_rows_per_state) if model.n_states else 0
    M = 2 * max_nnz + 2 * max_rows + int(dmax) + 2
    g = gamma(M, u)
    rho = lipschitz_rho(model)
    k = int(k)
    eta = k * g * (rho ** k) * ((1.0 + g) ** max(k - 1, 0)) if k > 0 else 0.0
    return FPErrorBound(eta_v=float(eta), M=int(M), u=u, gamma_M=g, rho=rho, k=k,
                        max_nnz_row=max_nnz, max_rows_per_state=max_rows, dtype=dtype)

from __future__ import annotations

import ctypes as ct
import hashlib
import os
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
E3_DIR = REPO / "rebuttal" / "engine" / "e3"
LIB_PATH = E3_DIR / "build" / "libcubcr_gpu_e3.so"
COPY_SRC = E3_DIR / "cubcr_gpu.cu"
PAPER_SRC = REPO / "csrc" / "cubcr_gpu.cu"

_lib = None


def _stats_struct():
    from cubcr.engine_gpu import GPUStats
    return GPUStats


def lib():

    global _lib
    if _lib is None:
        from cubcr.engine_cpu import CModel
        L = ct.CDLL(str(LIB_PATH))

        L.cubcr_gpu_create.restype = ct.c_void_p
        L.cubcr_gpu_create.argtypes = [ct.POINTER(CModel), ct.c_int]
        L.cubcr_gpu_destroy.argtypes = [ct.c_void_p]
        L.cubcr_gpu_solve.restype = ct.c_int
        L.cubcr_gpu_solve.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_int, ct.c_int64,
                                      ct.c_void_p, ct.c_int, ct.c_int64, ct.c_int, ct.c_int, ct.c_int64]
        L.cubcr_gpu_last_stats.argtypes = [ct.c_void_p, ct.POINTER(_stats_struct())]
        L.cubcr_gpu_last_error.restype = ct.c_char_p
        L.cubcr_gpu_mem_info.argtypes = [ct.c_int, ct.POINTER(ct.c_size_t), ct.POINTER(ct.c_size_t)]
        L.cubcr_gpu_device_name.argtypes = [ct.c_int, ct.c_char_p, ct.c_int]
        L.cubcr_gpu_device_count.restype = ct.c_int

        D = ct.POINTER(ct.c_double)
        L.e3_context_init.restype = ct.c_int
        L.e3_context_init.argtypes = [ct.c_int, D]
        L.e3_host_build.restype = ct.c_void_p
        L.e3_host_build.argtypes = [ct.POINTER(CModel), D]
        L.e3_host_free.argtypes = [ct.c_void_p]
        L.e3_host_info.argtypes = [ct.c_void_p, ct.POINTER(ct.c_int64)]
        L.e3_alloc.restype = ct.c_void_p
        L.e3_alloc.argtypes = [ct.c_void_p, ct.POINTER(CModel), ct.c_int, D]
        L.e3_upload.restype = ct.c_int
        L.e3_upload.argtypes = [ct.c_void_p, ct.c_void_p, ct.POINTER(CModel), D, ct.POINTER(ct.c_int64)]
        L.e3_update_policy.restype = ct.c_int
        L.e3_update_policy.argtypes = [ct.c_void_p, ct.c_void_p, ct.POINTER(CModel), D]
        L.e3_update_values.restype = ct.c_int
        L.e3_update_values.argtypes = [ct.c_void_p, ct.c_void_p, ct.POINTER(CModel), D]
        L.e3_omp_max_threads.restype = ct.c_int
        _lib = L
    return _lib


def _err() -> str:
    return lib().cubcr_gpu_last_error().decode()


def sha256(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def provenance() -> dict:
    return {"e3_lib": str(LIB_PATH.relative_to(REPO)), "e3_lib_sha256": sha256(LIB_PATH),
            "copied_source_sha256": sha256(COPY_SRC), "paper_source_sha256": sha256(PAPER_SRC),
            "copy_is_verbatim": sha256(COPY_SRC) == sha256(PAPER_SRC),
            "e3_api_sha256": sha256(E3_DIR / "e3_api.cu")}


def context_init(device: int = 0) -> float:

    ms = ct.c_double(0.0)
    rc = lib().e3_context_init(device, ct.byref(ms))
    if rc:
        raise RuntimeError("e3_context_init failed: " + _err())
    return ms.value / 1e3


def mem_info(device: int = 0):
    f, t = ct.c_size_t(), ct.c_size_t()
    if lib().cubcr_gpu_mem_info(device, ct.byref(f), ct.byref(t)):
        raise RuntimeError(_err())
    return int(f.value), int(t.value)


def omp_max_threads() -> int:
    return int(lib().e3_omp_max_threads())


def same_structure(a, b) -> bool:

    return (a.n_states == b.n_states and a.init_state == b.init_state
            and np.array_equal(a.state_kind, b.state_kind) and np.array_equal(a.state_row_start, b.state_row_start)
            and np.array_equal(a.state_agent, b.state_agent) and np.array_equal(a.state_radix, b.state_radix))


def same_sparsity(a, b) -> bool:

    return (same_structure(a, b) and np.array_equal(a.row_ptr, b.row_ptr) and np.array_equal(a.col, b.col)
            and np.array_equal(a.weights, b.weights))


class E3Engine:


    def __init__(self, compiled, device: int = 0, dtype: str = "fp64", use_graph: bool = True, max_batch: int = 0,
                 tpb: int = 0, l2_tile: int = 0, create: str = "phased", defer_device: bool = False):


        if l2_tile and l2_tile > 0 and use_graph:
            raise ValueError("l2_tile > 0 with use_graph=True is not supported (as in cubcr.engine_gpu)")
        self.device, self.dtype, self.use_graph = device, dtype, use_graph
        self.max_batch, self.tpb, self.l2_tile = max_batch, tpb, l2_tile
        self.compiled = None
        self._ctx = None
        self._host = None
        self._m = None
        self.create_mode = create
        self.stats = None
        self.phase_s = {}
        self.calls = []
        self.reset_cum()
        self._create(compiled, defer_device=defer_device)


    def _create(self, comp, defer_device: bool = False):
        from cubcr.engine_cpu import model_struct
        L = lib()
        ph = {}
        t0 = time.perf_counter()
        m, keep = model_struct(comp)
        ph["marshal_s"] = time.perf_counter() - t0
        self.compiled = comp
        if self.create_mode == "paper":
            t0 = time.perf_counter()
            ctx = L.cubcr_gpu_create(ct.byref(m), self.device)
            ph["create_paper_s"] = time.perf_counter() - t0
            if not ctx:
                raise RuntimeError("cubcr_gpu_create failed: " + _err())
            self._ctx, self._host = ctx, None
            self.phase_s = ph
            return ph
        if self.create_mode != "phased":
            raise ValueError(self.create_mode)
        ms2 = (ct.c_double * 4)()
        t0 = time.perf_counter()
        h = L.e3_host_build(ct.byref(m), ms2)
        ph["p2_host_s"] = time.perf_counter() - t0
        if not h:
            raise RuntimeError("e3_host_build failed: " + _err())
        ph["p2_c_tables_s"], ph["p2_c_nrow_s"], ph["p2_c_val32_s"], ph["p2_c_classes_s"] = [x / 1e3 for x in ms2]
        info = (ct.c_int64 * 5)()
        L.e3_host_info(h, info)
        ph["table_entries"], ph["table_bytes"], ph["rmax"], ph["n_row_classes"], ph["slist_len"] = [int(x) for x in info]
        self._host = h
        self._m = (m, keep)
        self.phase_s = ph
        if not defer_device:
            self.attach_device()
        return self.phase_s

    def attach_device(self) -> dict:

        if self._host is None or self._m is None or self._ctx is not None:
            raise RuntimeError("attach_device: host phase missing or device already attached")
        L = lib()
        m, keep = self._m
        ph = self.phase_s
        ms3 = (ct.c_double * 1)()
        t0 = time.perf_counter()
        ctx = L.e3_alloc(self._host, ct.byref(m), self.device, ms3)
        ph["p3_alloc_s"] = time.perf_counter() - t0
        if not ctx:
            raise RuntimeError("e3_alloc failed: " + _err())
        ph["p3_c_alloc_s"] = ms3[0] / 1e3
        ms4 = (ct.c_double * 2)()
        nbytes = ct.c_int64(0)
        t0 = time.perf_counter()
        rc = L.e3_upload(ctx, self._host, ct.byref(m), ms4, ct.byref(nbytes))
        ph["p4_upload_s"] = time.perf_counter() - t0
        if rc:
            L.cubcr_gpu_destroy(ctx)
            raise RuntimeError("e3_upload failed: " + _err())
        ph["p4_c_upload_s"], ph["p4_event_s"] = ms4[0] / 1e3, ms4[1] / 1e3
        ph["h2d_bytes"] = int(nbytes.value)
        self._ctx = ctx
        self._m = None
        del keep
        return ph

    def close(self):
        L = lib() if (self._ctx or self._host) else None
        if self._ctx:
            L.cubcr_gpu_destroy(self._ctx)
            self._ctx = None
        if self._host:
            L.e3_host_free(self._host)
            self._host = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


    def reset_cum(self):

        self.cum = {"calls": 0, "coalitions": 0, "total_ms": 0.0, "kernel_ms": 0.0, "waves": 0, "max_batch": 0, "py_s": 0.0}
        self.calls = []

    def solve(self, masks, k=None) -> np.ndarray:
        from cubcr.engine_cpu import prepare_masks
        t_py = time.perf_counter()
        if k is None:
            k = self.compiled.horizon_k
        arr, stride, B = prepare_masks(masks, k)
        out = np.empty(B, dtype=np.float64)
        L = lib()
        rc = L.cubcr_gpu_solve(self._ctx, int(k), arr.ctypes.data, int(stride), int(B), out.ctypes.data,
                               0 if self.dtype == "fp64" else 1, int(self.max_batch), 1 if self.use_graph else 0,
                               int(self.tpb), int(self.l2_tile))
        if rc != 0:
            raise RuntimeError(f"cubcr_gpu_solve failed rc={rc}: {_err()}")
        st = _stats_struct()()
        L.cubcr_gpu_last_stats(self._ctx, ct.byref(st))
        self.stats = st.as_dict()
        py_s = time.perf_counter() - t_py
        c = self.cum
        c["calls"] += 1; c["coalitions"] += int(B); c["total_ms"] += float(self.stats["total_ms"])
        c["kernel_ms"] += float(self.stats["kernel_ms"]); c["waves"] += int(self.stats["waves"])
        c["max_batch"] = max(c["max_batch"], int(self.stats["batch"])); c["py_s"] += py_s
        self.calls.append({"B": int(B), "total_ms": float(self.stats["total_ms"]), "kernel_ms": float(self.stats["kernel_ms"]),
                           "batch": int(self.stats["batch"]), "waves": int(self.stats["waves"]),
                           "graph_captures": int(self.stats["graph_captures"]), "py_s": py_s})
        return out


    def update_policy(self, new_comp) -> dict:

        if self._host is None:
            raise RuntimeError("update paths need create='phased'")
        from cubcr.engine_cpu import model_struct
        t_all = time.perf_counter()
        t0 = time.perf_counter()
        if not same_structure(self.compiled, new_comp):
            raise ValueError("R1 update needs an unchanged structure (kind/row_start/agent/radix); use rebuild()")
        if not (np.array_equal(self.compiled.row_ptr, new_comp.row_ptr) and np.array_equal(self.compiled.col, new_comp.col)
                and np.array_equal(self.compiled.val, new_comp.val)):
            raise ValueError("R1 update: transitions changed; use update_values() / rebuild()")
        check_s = time.perf_counter() - t0
        m, keep = model_struct(new_comp)
        ms = (ct.c_double * 3)()
        t0 = time.perf_counter()
        rc = lib().e3_update_policy(self._ctx, self._host, ct.byref(m), ms)
        call_s = time.perf_counter() - t0
        if rc:
            raise RuntimeError(f"e3_update_policy failed rc={rc}: {_err()}")
        del keep
        self.compiled = new_comp
        return {"check_s": check_s, "update_call_s": call_s, "host_recompute_s": ms[0] / 1e3, "upload_s": ms[1] / 1e3,
                "upload_event_s": ms[2] / 1e3, "total_s": time.perf_counter() - t_all,
                "bytes": int(self.phase_s.get("table_bytes", 0))}

    def update_values(self, new_comp) -> dict:

        if self._host is None:
            raise RuntimeError("update paths need create='phased'")
        from cubcr.engine_cpu import model_struct
        t_all = time.perf_counter()
        t0 = time.perf_counter()
        if not (same_structure(self.compiled, new_comp) and np.array_equal(self.compiled.row_ptr, new_comp.row_ptr)
                and np.array_equal(self.compiled.col, new_comp.col)):
            raise ValueError("R2 update needs an unchanged sparsity pattern; use rebuild()")
        if not np.array_equal(self.compiled.weights, new_comp.weights):
            raise ValueError("R2 update: policies changed as well; call update_policy() too")
        check_s = time.perf_counter() - t0
        m, keep = model_struct(new_comp)
        ms = (ct.c_double * 3)()
        t0 = time.perf_counter()
        rc = lib().e3_update_values(self._ctx, self._host, ct.byref(m), ms)
        call_s = time.perf_counter() - t0
        if rc:
            raise RuntimeError(f"e3_update_values failed rc={rc}: {_err()}")
        del keep
        self.compiled = new_comp
        return {"check_s": check_s, "update_call_s": call_s, "host_fp32_s": ms[0] / 1e3, "upload_s": ms[1] / 1e3,
                "upload_event_s": ms[2] / 1e3, "total_s": time.perf_counter() - t_all,
                "bytes": int(new_comp.nnz * 12)}

    def rebuild(self, new_comp) -> dict:

        t_all = time.perf_counter()
        t0 = time.perf_counter()
        self.close()
        destroy_s = time.perf_counter() - t0
        self.create_mode = "phased"
        ph = self._create(new_comp)
        out = {"destroy_s": destroy_s, "total_s": time.perf_counter() - t_all}
        out.update({k: v for k, v in ph.items()})
        return out

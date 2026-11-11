#!/usr/bin/env python3


import os as _os_anon
_CUBCR_REPO = _os_anon.environ.get("CUBCR_ROOT") or _os_anon.path.abspath(_os_anon.path.join(
    _os_anon.path.dirname(_os_anon.path.abspath(__file__)), "../.."))
import json
import sys
import time

import numpy as np

sys.path.insert(0, (_CUBCR_REPO + "/rebuttal/lib"))
import rb
import e3_engine as E
import e3_common as C

OUT = C.RAW / "create_paths"
CONFIGS = ["n32_S1e4_k100", "dt_n12_S1e5_k25", "oc_k10"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [c for c in CONFIGS if not (OUT / f"{c}.json").exists()]
    built = {c: C.build(C.config(c)) for c in todo}
    if not todo:
        print("all present")
        return 0
    with rb.gpu_lock("E3 create-path check") as waited:
        E.context_init(0)
        hw = rb.hardware_tags()
        for c in todo:
            comp = built[c][0]
            snap = rb.load_snapshot()
            e = rb.make_gpu_engine(comp); e.close()
            e = E.E3Engine(comp); e.close()
            rows = []
            for i in range(5):
                for mode in ("submitted", "e3_verbatim", "e3_phased"):
                    t0 = time.perf_counter()
                    if mode == "submitted":
                        e = rb.make_gpu_engine(comp)
                    elif mode == "e3_verbatim":
                        e = E.E3Engine(comp, create="paper")
                    else:
                        e = E.E3Engine(comp)
                    dt = time.perf_counter() - t0
                    ph = {k: v for k, v in (getattr(e, "phase_s", None) or {}).items()}
                    e.close()
                    rows.append({"rep": i + 1, "mode": mode, "create_s": dt, **ph})
            rec = {"config": c, "states": int(comp.n_states), "nnz": int(comp.nnz), "rows": rows, "gpu_lock_wait_s": waited,
                   "load_before": snap, "load_after": rb.load_snapshot(), "hardware": hw, "e3_provenance": E.provenance(),
                   "omp_max_threads": E.omp_max_threads(), "timestamp": rb.now_iso()}
            rb.write_json(OUT / f"{c}.json", rec)
            med = {m: float(np.median([r["create_s"] for r in rows if r["mode"] == m])) for m in ("submitted", "e3_verbatim", "e3_phased")}
            print(f"[{rb.now_iso()}] {c}: median create  " + "  ".join(f"{k} {v * 1e3:.1f} ms" for k, v in med.items()), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

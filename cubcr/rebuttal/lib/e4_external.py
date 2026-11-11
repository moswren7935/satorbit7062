from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np

EXT = Path(os.environ.get("CUBCR_EXTERNAL_DIR", Path(__file__).resolve().parents[1] / "env" / "ext"))
LEVERAGE_SHA = "9702c2cc6386e13fca20b9caeb8747dedf253cef"
ADALINA_SHA = "19554d00e8fe9338c43e66ec439eb5c4ab747135"
SVARM_SHA = "c4e30395b14b8bf9a8687d839d5b406f825bb23d"
LEVERAGE_DIR = EXT / f"leverageshap-{LEVERAGE_SHA}"
ADALINA_DIR = EXT / f"adalina-{ADALINA_SHA}"
SVARM_DIR = EXT / f"svarm-{SVARM_SHA}"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_LEV = None
_ADA = None


def leverage_module():

    global _LEV
    if _LEV is not None:
        return _LEV
    root = LEVERAGE_DIR / "leverageshap"
    if not root.exists():
        raise FileNotFoundError(f"{root} missing: run rebuttal/scripts/e4_fetch_external.sh")
    pkg = types.ModuleType("leverageshap"); pkg.__path__ = [str(root)]
    sys.modules.setdefault("leverageshap", pkg)
    ut = types.ModuleType("leverageshap.utils")

    class Game:
        pass
    ut.Game = Game
    sys.modules.setdefault("leverageshap.utils", ut)
    est = types.ModuleType("leverageshap.estimators"); est.__path__ = [str(root / "estimators")]
    sys.modules.setdefault("leverageshap.estimators", est)
    _load("leverageshap.estimators.sampling", root / "estimators" / "sampling.py")
    _LEV = _load("leverageshap.estimators.leverage_shap", root / "estimators" / "leverage_shap.py")
    return _LEV


def adalina_modules():

    global _ADA
    if _ADA is not None:
        return _ADA
    if not ADALINA_DIR.exists():
        raise FileNotFoundError(f"{ADALINA_DIR} missing: run rebuttal/scripts/e4_fetch_external.sh")
    pkg = types.ModuleType("e4ext_adalina_estimators"); pkg.__path__ = [str(ADALINA_DIR / "estimators")]
    sys.modules.setdefault("e4ext_adalina_estimators", pkg)
    mod = _load("e4ext_adalina_estimators.Adalina", ADALINA_DIR / "estimators" / "Adalina.py")
    ut = _load("e4ext_adalina_utilTemplate", ADALINA_DIR / "utilFuncs" / "utilTemplate.py")
    _ADA = (mod.Adalina, ut.utilTemplate)
    return _ADA


class LeverageGameAdapter:


    def __init__(self, value_of_masks, d: int):
        self.value_of_masks = value_of_masks
        self.d = d
        self.w = (np.int64(1) << np.arange(d, dtype=np.int64))

    def value(self, S):
        S = np.asarray(S).astype(bool).reshape(-1, self.d)
        return self.value_of_masks((S.astype(np.int64) * self.w[None, :]).sum(axis=1))

    def edge_cases(self):
        v = self.value_of_masks(np.array([0, (1 << self.d) - 1], dtype=np.int64))
        return float(v[0]), float(v[1])


def run_leverage_shap(value_of_masks, d: int, budget: int, seed: int) -> np.ndarray:
    mod = leverage_module()
    est = mod.LeverageSHAP(d, LeverageGameAdapter(value_of_masks, d), paired_sampling=True, random_state=int(seed))
    return np.asarray(est.shap_values(int(budget)), dtype=np.float64).reshape(-1)


def run_adalina(value_of_masks, d: int, budget: int, seed: int, batch: int = 100) -> np.ndarray:


    Adalina, utilTemplate = adalina_modules()
    q = (int(budget) - 2) // d
    if q < 1:
        raise ValueError("budget below Adalina's minimum (d + 2)")
    w = (np.int64(1) << np.arange(d, dtype=np.int64))

    class TableUtility(utilTemplate):
        def __init__(self):
            self.n_players = d
            self.weights = None

        def evaluate(self, subset):
            m = int(np.sum(w[np.asarray(subset, dtype=bool)]))
            return float(value_of_masks(np.array([m], dtype=np.int64))[0])

    util = TableUtility()
    est = Adalina(util, (1, 1), q, q, batch)
    for b in est.batch_generator(int(seed)):
        est.manage_raw_results(est.process_batch(b))
    return np.asarray(est.estimate_traj[-1], dtype=np.float64).reshape(-1)

import argparse
import importlib.util
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
STAGES = ("preparation", "primary", "engine", "tables", "decision", "overcooked", "validation",
          "E1", "E2", "E3", "E4", "E5", "E7", "E9", "report")


def commands(stage, workers):
    def py(name, *args):
        return [sys.executable, "rebuttal/scripts/" + name + ".py", *map(str, args)]

    def original(name, *args):
        return [sys.executable, "experiments/" + name + ".py", *map(str, args)]

    if stage == "preparation":
        return [["make", "-C", "csrc", "all"],
                ["bash", "rebuttal/scripts/e4_fetch_external.sh"]]
    if stage == "primary":
        return [original("e9v3_certified", "--scope", "full"),
                original("e9v3_certified", "--genuine-to")]
    if stage == "engine":
        return [original("grid", "e1"), original("grid", "e2"), original("grid", "e1prune")]
    if stage == "tables":
        return [original("grid", "e0", "--n-list", 12, 16, "--states-list", 1000, 10000,
                         "--k-list", 25, 100),
                original("grid", "e0hz"), original("e11_bakeoff", "--tables-only"), py("gen_tables")]
    if stage == "decision":
        return [original("e4_sampling", "--value-range", "monotone", "--ci", "prpl_eb_cs",
                         "--batch-schedule", "geometric", "--exp", "e4v2", "--out", "results/e4v2",
                         "--time-budget", 180, "--max-perms", 100000)]
    if stage == "overcooked":
        return [original("e14_overcooked", "--models"), py("e2a_gpu_runs", "--overcooked-table")]
    if stage == "validation":
        return [[sys.executable, "datasets/fetch.py", "--skip-verify"],
                [sys.executable, "generators/overcooked_csg.py", "--k", "20", "30", "40"],
                original("e14_overcooked", "--skip-exact"),
                original("e11_bakeoff", "--arms", "ours_cs", "--skip-existing"),
                original("e12_loo", "--skip-existing"),
                original("e3_prism", "--bench-file", "e3b", "--prism-extra", "-javastack 1g"),
                original("e3storm", "--configs", "full")]
    if stage == "E1":
        return [py("e1_validate_inputs", "--workers", workers),
                py("e1_precision_audit", "--phase", "all", "--workers", workers),
                py("e1_conservative_intervals", "--phase", "gpu"),
                py("e1_outward_bound", "--workers", workers),
                py("e1_conservative_intervals", "--phase", "cpu", "--workers", workers),
                py("e1_conservative_intervals", "--phase", "summary")]
    if stage == "E2":
        return [py("e2a_oracle_runs", "--divs", 20, 50, 100, "--seedsets", "paper", "new",
                   "--max-n", 24, "--workers", workers),
                py("e2a_gpu_runs", "--n28", "--divs", 20, 50, 100, "--seedsets", "paper", "new"),
                py("e2b_grid", "--divs", 10, 20, 50, 100, "--allow-rebuilt-engine"),
                py("e2b_dt", "--radii", 0.005, 0.001, "--skip-big", "--allow-rebuilt-engine"),
                py("e2b_dt", "--radii", 0.005, "--models", 5, "--allow-rebuilt-engine"),
                py("e2b_dt_big", "--eps", 0.001, "--allow-rebuilt-engine"),
                py("e2b_overcooked", "--divs", 20, 100, "--allow-rebuilt-engine")]
    if stage == "E3":
        return [["make", "-C", "rebuttal/engine/e3", "all"],
                py("e3_timing_phases", "--reps", 5), py("e3_revisions", "--reps", 5),
                py("e3_create_paths")]
    if stage == "E4":
        return [py("e4_prepare_games"), py("e4_run_grid", "--workers", workers),
                py("e4_run_adaptive", "--workers", workers, "--eps-divs", 20, 100, "--seeds", 30),
                py("e4_run_equal_budget", "--workers", workers, "--divs", 20, 100)]
    if stage == "E5":
        return [["make", "-C", "rebuttal/engine/e5", "all"],
                py("e5_measure", "--engine", "paper", "--configs", "1,8", "3,2", "2,4", "2,8", "3,4",
                   "--size", "1e3", "--B", 65536, "--reps", 5),
                py("e5_measure", "--engine", "paper", "--configs", "1,8", "3,2", "2,4", "2,8", "3,4",
                   "--size", "1e4", "--B", 65536, "--reps", 3),
                py("e5_measure", "--engine", "acc", "--configs", "4,4", "3,8",
                   "--size", "1e3", "--B", 65536, "--reps", 5)]
    if stage == "E7":
        return [py("e7_gpu_runs", "--phase", "topk3,topk1", "--min-n", 16, "--max-n", 28)]
    if stage == "E9":
        return [py("e9_plan"), py("e9_run"), py("e9_run_hostlock", "--no-gpu-test"),
                py("e9_twins", "--workers", workers)]
    if stage == "report":
        return [original("revised_paper")]
    raise ValueError(stage)


def require_files(paths, stage):
    missing = [str(p.relative_to(ROOT)) for p in paths if not p.is_file()]
    if missing:
        detail = ", ".join(missing[:4])
        if len(missing) > 4:
            detail += f", and {len(missing) - 4} more"
        raise SystemExit(f"Missing inputs: {detail}. Run --stage {stage} first.")


def require_primary():
    sys.path.insert(0, str(ROOT / "rebuttal/lib"))
    import rb
    try:
        tasks = rb.paper_tasks()
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    expected = {(n, s, k, seed) for n in rb.GRID_N for s in rb.GRID_STATES
                for k in rb.GRID_K for seed in rb.PAPER_GEN_SEEDS}
    found = {(t["n"], t["states_target"], t["k"], t["gen_seed"]) for t in tasks}
    if found != expected or any(
            not all(seed in t["cert_gpu"] for seed in rb.PAPER_SAMPLER_SEEDS) or
            t["exact_gpu"] is None or t["exact_cpu"] is None for t in tasks):
        raise SystemExit("Incomplete primary grid; run --stage primary first.")
    return tasks


def require_table_inputs():
    for family, count in (("e0", 8), ("e0hz", 4), ("e11", 1)):
        if len(list((ROOT / "results" / family / "values").glob("*.npy"))) < count:
            raise SystemExit("Missing exact-game tables; run --stage tables first.")


def require_overcooked_data():
    spec = importlib.util.find_spec("overcooked_ai_py")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("Overcooked requires the overcooked-ai package from requirements.txt.")
    data = Path(next(iter(spec.submodule_search_locations))) / "data/human_data/clean_train_trials.pickle"
    if not data.is_file():
        raise SystemExit(f"Overcooked requires its external human training data at {data}.")


def prerequisites(stage):
    if stage in ("preparation", "report"):
        return
    require_files([ROOT / "csrc/build" / f"libcubcr_{engine}.so" for engine in ("cpu", "gpu")], "preparation")
    if stage in ("tables", "validation", "E1", "E2", "E3", "E4", "E7", "E9"):
        tasks = require_primary()
        if stage in ("validation", "E1", "E2", "E4", "E9"):
            require_files([ROOT / "rebuttal/cache/tables" / f"{t['iid']}.npy"
                           for t in tasks if t["n"] <= 24], "tables")
    if stage in ("validation", "E1", "E4"):
        require_table_inputs()
    if stage in ("overcooked", "validation", "E1", "E2", "E3", "E4"):
        require_overcooked_data()
    if stage in ("validation", "E1", "E2", "E3", "E4"):
        prefix = ROOT / "results/e14"
        require_files([prefix / "e14__overcooked_cramped_room_k10__exact_step.json"] +
                      [prefix / f"e14__overcooked_cramped_room_k10__cert_gpu_eps_vs_exact_s{s}.json"
                       for s in (20260829, 20260830, 20260831)], "overcooked")
    if stage in ("validation", "E1", "E2", "E4"):
        require_files([ROOT / "rebuttal/cache/tables/overcooked_k10_step.npy"], "overcooked")
    if stage in ("validation", "E1", "E2", "E3"):
        if len(list((ROOT / "results/e4v2").glob("e4__*__eps0.005__s20260829.json"))) < 6:
            raise SystemExit("Missing decision-time reference runs; run --stage decision first.")
    if stage in ("E4", "E9"):
        ext = Path(os.environ.get("CUBCR_EXTERNAL_DIR", ROOT / "rebuttal/env/ext"))
        for name, revision in (("leverageshap", "9702c2cc6386e13fca20b9caeb8747dedf253cef"),
                               ("adalina", "19554d00e8fe9338c43e66ec439eb5c4ab747135"),
                               ("svarm", "c4e30395b14b8bf9a8687d839d5b406f825bb23d")):
            if not (ext / f"{name}-{revision}").is_dir():
                raise SystemExit("Missing external estimators; run --stage preparation first.")
    if stage == "E9":
        require_files([ROOT / "rebuttal/results/E4/raw/equal_budget_plan.csv"], "E4")
        if not any((ROOT / "rebuttal/results/E4/raw/equal_budget").glob("*.csv.gz")):
            raise SystemExit("Missing E4 equal-budget runs; run --stage E4 first.")
    if stage in ("validation", "E1"):
        sys.path.insert(0, str(ROOT))
        from cubcr import io_prism
        if not io_prism.prism_available() or not shutil.which("java", path=io_prism.prism_env().get("PATH")):
            raise SystemExit("Validation needs PRISM-games and Java; set CUBCR_PRISM_BIN to its executable.")
        from cubcr.io_storm import resolve_bin
        if not shutil.which(resolve_bin()):
            raise SystemExit("Validation needs Storm; set CUBCR_STORM_BIN to its executable.")
    if stage == "E1":
        require_files([ROOT / "datasets/overcooked" / f"overcooked_cramped_room_k{k}.npz"
                       for k in (20, 30, 40)] +
                      [ROOT / "results/e14" / f"e14__overcooked_cramped_room_k{k}__agent_exact.json"
                       for k in (20, 30, 40)], "validation")
        for family, pattern, count in (("e12", "e12__ticketbench_new__*.json", 90),
                                       ("e3b", "e3__*.json", 9), ("e3storm", "e3storm__*.json", 6),
                                       ("e11", "*__ours_cs__*.json", 312)):
            if len(list((ROOT / "results" / family).glob(pattern))) < count:
                raise SystemExit(f"Missing {family} reference runs; run --stage validation first.")
        sys.path.insert(0, str(ROOT))
        from experiments.e3b_bench import active_bench
        if any(not Path(item[1]).is_file() for item in active_bench()):
            raise SystemExit("Missing PRISM model inputs; run --stage validation first.")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Generate fresh baseline and selected supplemental experiments from source.",
        epilog=("Bootstrap: preparation primary tables decision overcooked. Engine benchmarks: engine. "
                "Then run E2 E3 E4 E5 E7 E9; "
                "E9 needs E4. Optional validation then E1 also needs PRISM-games/Java and Storm "
                "(CUBCR_PRISM_BIN, CUBCR_STORM_BIN). Overcooked needs external human training data. "
                "Run report last to generate tables and figures. "
                "Full runs are expensive and require CUDA; generated outputs stay local. "
                "Existing outputs are reused where supported. See README.md."))
    ap.add_argument("--stage", choices=STAGES, nargs="+", required=True,
                    help="stages run in dependency order; prerequisites are not started implicitly")
    ap.add_argument("--dry-run", action="store_true", help="print commands only; no subprocesses or input checks")
    ap.add_argument("--workers", type=int, choices=range(1, 7), default=4, help="CPU worker count (default: 4)")
    args = ap.parse_args(argv)
    env = dict(os.environ, CUBCR_ROOT=str(ROOT))
    for stage in STAGES:
        if stage not in args.stage:
            continue
        print(f"[{stage}]", flush=True)
        if not args.dry_run:
            prerequisites(stage)
        for command in commands(stage, args.workers):
            print(shlex.join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=ROOT, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import rb


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate exact primary-game tables after the primary stage.")
    ap.parse_args(argv)
    tasks = [t for t in rb.paper_tasks() if t["n"] <= 24]
    expected = {(n, s, k, seed) for n in (16, 20, 24) for s in rb.GRID_STATES
                for k in rb.GRID_K for seed in rb.PAPER_GEN_SEEDS}
    found = {(t["n"], t["states_target"], t["k"], t["gen_seed"]) for t in tasks}
    if found != expected:
        raise SystemExit("Incomplete primary grid; run python experiments/rerun_revised.py --stage primary")
    for task in tasks:
        if rb.table_path(task["iid"]).exists():
            print("exists", task["iid"], flush=True)
            continue
        start = time.perf_counter()
        rb.exact_table(task)
        print(f"done {task['iid']} {time.perf_counter() - start:.1f}s", flush=True)
    print(f"{len(tasks)} primary tables available", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

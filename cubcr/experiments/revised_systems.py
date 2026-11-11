from __future__ import annotations

import csv
import gzip
import json
import math
import struct
from collections import defaultdict
from contextvars import ContextVar
from pathlib import Path
from statistics import mean, median


SEEDS = set(range(20260829, 20260859))
INPUT_FILES = ContextVar("systems_input_files", default=None)
ARMS = {"PP": "cubcr_paired_perm", "2-ADD": "shapiq_kADDSHAP", "Leverage": "LeverageSHAP", "Stratified": "StratifiedSVARM", "UKS": "shapiq_UnbiasedKernelSHAP"}
BUDGETS = {
    12: {48, 68, 94, 134, 188, 264, 372, 524, 736, 1036, 1456, 2048},
    16: {64, 112, 198, 350, 618, 1090, 1922, 3390, 5978, 10540, 18584, 32768},
    20: {80, 148, 270, 498, 918, 1688, 3106, 5716, 10518, 19354, 35614, 65536},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(root, relative):
    value = json.loads((root / relative).read_text())
    if INPUT_FILES.get() is not None:
        INPUT_FILES.get().add(Path(relative).as_posix())
    return value


def csv_rows(root, relative):
    path = root / relative
    with (gzip.open if path.suffix == ".gz" else open)(path, "rt", newline="") as handle:
        if INPUT_FILES.get() is not None:
            INPUT_FILES.get().add(Path(relative).as_posix())
        yield from csv.DictReader(handle)


def csv_chunks(root, relative):
    for path in sorted((root / relative).glob("*.csv.gz")):
        yield from csv_rows(root, path.relative_to(root))


def json_rows(root, relative):
    for path in sorted((root / relative).glob("*.json")):
        yield read_json(root, path.relative_to(root))


def groups(rows, fields):
    result = defaultdict(list)
    for row in rows:
        result[tuple(row[field] for field in fields)].append(row)
    return result


def indexed(rows, fields, label):
    result = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        require(key not in result, f"{label}: duplicate {key}")
        result[key] = row
    return result


def extent(values):
    values = list(values)
    require(bool(values) and all(math.isfinite(x) for x in values), "Empty or nonfinite aggregate")
    return {"min": min(values), "median": median(values), "max": max(values), "n": len(values)}


def close(actual, expected, label):
    require(math.isclose(float(actual), float(expected), rel_tol=2e-11, abs_tol=2e-14), f"{label}: {actual} != {expected}")


def vector(text):
    return [float(value) for value in text.split()]


def errors(row, game):
    estimate = vector(row["phi_hat"])
    reference = vector(row["phi_exact"]) if row.get("phi_exact") else game["phi_exact"]
    require(len(estimate) == len(reference) == game["d"], f"Vector length: {row['table_id']}")
    require(all(math.isfinite(x) for x in estimate + reference), "Nonfinite attribution vector")
    for value, expected in zip(reference, game["phi_exact"]):
        close(value, expected, "Exact attribution vector")
    differences = [a - b for a, b in zip(estimate, reference)]
    maximum = max(abs(value) for value in differences)
    mse = mean(value * value for value in differences)
    close(maximum, row["max_abs"], "Stored maximum error")
    close(mse, row["mse"], "Stored mean squared error")
    return maximum, mse


def quantile95(values):
    values = sorted(values)
    position = .95 * (len(values) - 1)
    left = int(position)
    return values[left] + (position - left) * (values[min(left + 1, len(values) - 1)] - values[left])


def crossing(points, epsilon):
    points = sorted(points, key=lambda pair: pair[0])
    violating = [i for i, (_, error) in enumerate(points) if error > epsilon]
    if not violating:
        return points[0][0], "le_min"
    index = violating[-1]
    if index == len(points) - 1:
        return points[-1][0], "gt_max"
    (x0, y0), (x1, y1) = points[index:index + 2]
    if y1 <= 0:
        return x1, "ok"
    fraction = min(1., max(0., (math.log(epsilon) - math.log(y0)) / (math.log(y1) - math.log(y0))))
    return math.exp(math.log(x0) + fraction * (math.log(x1) - math.log(x0))), "ok"


def censored_median(values):
    values = sorted(values, key=lambda pair: math.inf if pair[1] == "gt_max" else pair[0])
    n = len(values)
    middle = [values[(n - 1) // 2]] if n % 2 else values[n // 2 - 1:n // 2 + 1]
    flags = {flag for _, flag in middle}
    require(not {"gt_max", "le_min"} <= flags, "Median has incompatible censoring directions")
    qualifier = ">" if "gt_max" in flags else "<=" if "le_min" in flags else ""
    return {"budget": mean(value for value, _ in middle), "qualifier": qualifier}


def e3(root):
    configurations = {f"n{n}_S{size}_k{k}" for n in (16, 20, 24, 28, 32) for size in ("1e3", "1e4") for k in (50, 100)}
    all_configurations = configurations | {"oc_k10", "dt_n12_S1e5_k25"}
    phase_rows = []
    for record in json_rows(root, "rebuttal/results/E3/raw/phases"):
        if record["rep"] < 1:
            continue
        row = {key: record[key] for key in ("config", "mode", "rep")}
        row.update(imports=(record.get("interp_start_s") or 0.) + (record.get("imports_s") or 0.),
                   P0=record["build_p0_s"] + record["build_p0m_s"],
                   P1=record["build_p1_s"] + record["marshal_s"], P2=record["p2_s"],
                   P3a=record["p3a_context_s"], P3b=record["p3b_alloc_s"],
                   P4=record["p4_upload_s"], U=record["ups_solve_s"], P5=record["p5_s"],
                   P6=record["p6_assembly_s"], total=record["total_s"])
        row["build"] = row["P0"] + row["P1"] + row["P2"]
        phase_rows.append(row)
    indexed(phase_rows, ("config", "mode", "rep"), "E3 phases")
    phase_groups = groups(phase_rows, ("config", "mode"))
    require(set(phase_groups) == {(config, mode) for config in all_configurations for mode in ("cold", "warm")}, "Incomplete E3 phase configurations")
    phases = {}
    for key, rows in phase_groups.items():
        require(len(rows) == 5 and {int(row["rep"]) for row in rows} == set(range(1, 6)), f"E3 repetitions: {key}")
        for row in rows:
            close(sum(float(row[field]) for field in ("P0", "P1", "P2")), row["build"], "E3 construction phases")
            close(sum(float(row[field]) for field in ("imports", "P0", "P1", "P2", "P3a", "P3b", "P4", "U", "P5", "P6")), row["total"], "E3 complete first-result total")
        phases[key] = {"total": median(float(row["total"]) for row in rows), "construction": median(sum(float(row[field]) for field in ("P0", "P1", "P2")) for row in rows)}
    revision_rows = []
    for record in json_rows(root, "rebuttal/results/E3/raw/revisions"):
        if record.get("status") == "NOT APPLICABLE":
            continue
        row = {key: record[key] for key in ("config", "revision", "rep", "update_s", "ups_solve_s",
               "p5_s", "reattr_s", "p1_reassembly_s", "reattr_incl_p1_s", "fresh_create_submitted_s")}
        row.update({key: record["correctness"][key] for key in
                    ("endpoints_bitwise", "random_bitwise", "sampler_same_as_fresh")})
        row["revert_ok"] = record["revert_endpoints_bitwise"]
        revision_rows.append(row)
    indexed(revision_rows, ("config", "revision", "rep"), "E3 revisions")
    revision_groups = groups(revision_rows, ("config", "revision"))
    expected = {(config, "R1") for config in all_configurations}
    expected |= {(f"n{n}_S{size}_k50", rev) for n in (16, 28, 32) for size in ("1e3", "1e4") for rev in ("R2", "R3")}
    expected.add(("oc_k10", "R3"))
    require(set(revision_groups) == expected, "Incomplete E3 revision configurations")
    revisions = {}
    for key, rows in revision_groups.items():
        require(len(rows) == 5 and {int(row["rep"]) for row in rows} == set(range(1, 6)), f"E3 revision repetitions: {key}")
        for row in rows:
            require(all(row[field] is True for field in ("endpoints_bitwise", "random_bitwise", "revert_ok")), f"E3 update identity: {key}")
            require(row["sampler_same_as_fresh"] is None or row["sampler_same_as_fresh"] is True, f"E3 sampler identity: {key}")
            close(float(row["update_s"]) + float(row["ups_solve_s"]) + float(row["p5_s"]), row["reattr_s"], "E3 revision total")
            close(float(row["reattr_s"]) + float(row["p1_reassembly_s"]), row["reattr_incl_p1_s"], "E3 complete revision total")
        revisions[key] = {field: median(float(row[field]) for row in rows) for field in ("update_s", "fresh_create_submitted_s", "reattr_incl_p1_s")}
    policy = [revisions[config, "R1"] for config in sorted(all_configurations)]
    special = {}
    for name, config in (("overcooked", "oc_k10"), ("decision_time_300", "dt_n12_S1e5_k25")):
        special[name] = {"cold_s": phases[config, "cold"]["total"], "construction_s": phases[config, "cold"]["construction"], "warm_rebuild_s": phases[config, "warm"]["total"], "policy_reattribution_s": revisions[config, "R1"]["reattr_incl_p1_s"]}
    return {
        "phase_runs": len(phase_rows), "revision_runs": len(revision_rows), "revision_groups": len(revision_groups),
        "cold_grid_total_s": extent(phases[config, "cold"]["total"] for config in sorted(configurations)),
        "cold_grid_construction_s": extent(phases[config, "cold"]["construction"] for config in sorted(configurations)),
        "grid_policy_reattribution_s": extent(revisions[config, "R1"]["reattr_incl_p1_s"] for config in sorted(configurations)),
        "example_revision_s": {rev: revisions["n28_S1e3_k50", rev]["reattr_incl_p1_s"] for rev in ("R1", "R2", "R3")},
        "policy_update_ms": extent(row["update_s"] * 1000 for row in policy),
        "warm_create_minus_update_ms": extent((row["fresh_create_submitted_s"] - row["update_s"]) * 1000 for row in policy),
        "warm_create_faster_configurations": sum(row["fresh_create_submitted_s"] < row["update_s"] for row in policy),
        "complete_sampler_matches": sum(row["sampler_same_as_fresh"] is True for row in revision_rows), **special,
    }


def expected_game_ids():
    result = {"overcooked_k10_step"}
    for n, sizes in ((12, (1009, 10009)), (16, (961, 9985)), (20, (961, 9961))):
        for size in sizes:
            if n == 12:
                specifications = [(25, 20260829, ""), (100, 20260829, ""), (100, 20260829, "_hz0.02")]
            else:
                specifications = [(k, seed, "") for k in (50, 100) for seed in (20260829, 20260830, 20260831)]
                if n == 16:
                    specifications += [(25, 20260829, ""), (100, 20260829, "_hz0.02")]
            result.update(f"tb_none_n{n}_S{size}_k{k}_pf0.05_C3_s{seed}{suffix}" for k, seed, suffix in specifications)
    return result


def e4(root):
    games_list = read_json(root, "rebuttal/results/E4/raw/games.json")["games"]
    games = {key[0]: row for key, row in indexed(games_list, ("table_id",), "E4 games").items()}
    require(set(games) == expected_game_ids(), "Incomplete E4 game cohort")
    cells = defaultdict(list)
    for row in csv_chunks(root, "rebuttal/results/E4/raw/grid"):
        if row["estimator"] not in ARMS.values():
            continue
        require(row["table_id"] in games, "Unknown E4 game")
        cells[row["table_id"], row["estimator"], int(row["budget"])].append(row)
    expected = {(tid, arm, budget) for tid, game in games.items() for arm in ARMS.values() for budget in BUDGETS[game["d"]]}
    require(set(cells) == expected, "Incomplete E4 estimator/budget cohort")
    curves = defaultdict(list)
    for (tid, arm, budget), rows in sorted(cells.items()):
        require(len(rows) == 30 and {int(row["seed"]) for row in rows} == SEEDS, f"E4 seed cohort: {tid}/{arm}/{budget}")
        require(all(int(row["d"]) == games[tid]["d"] for row in rows), "E4 player count")
        statuses = {row["status"] for row in rows}
        require(statuses in ({"OK"}, {"INFEASIBLE"}), f"Unexpected E4 statuses: {statuses}")
        if statuses == {"INFEASIBLE"}:
            require(all(row["min_feasible_budget"] and budget < float(row["min_feasible_budget"]) for row in rows), f"E4 infeasible budget: {tid}/{arm}/{budget}")
            continue
        maximum_errors = [errors(row, games[tid])[0] for row in rows]
        curves[tid, arm].append((median(float(row["n_unique"]) for row in rows), quantile95(maximum_errors)))
    require(set(curves) == {(tid, arm) for tid in games for arm in ARMS.values()}, "E4 missing feasible curve")
    crossings = {(tid, arm, divisor): crossing(points, games[tid]["upsilon"] / divisor) for (tid, arm), points in curves.items() for divisor in (20, 100)}
    table4 = []
    for family, ids in (("TicketBench", sorted(tid for tid, game in games.items() if game["d"] == 20 and game["family"] != "overcooked")), ("Overcooked", ["overcooked_k10_step"])):
        require(len(ids) == (12 if family == "TicketBench" else 1), "E4 Table 4 game count")
        for divisor in (20, 100):
            table4.append({"game": family, "target": f"Upsilon/{divisor}", **{label: censored_median([crossings[tid, arm, divisor] for tid in ids]) for label, arm in ARMS.items()}})
    pair_rows = [row for row in csv_chunks(root, "rebuttal/results/E4/raw/equal_budget") if row["estimator"] in ("cubcr_paired_perm", "plain_perm") and row["eps_div"] == "20" and row["acct"] == "req" and row["d"] == "20" and row["family"] != "overcooked"]
    pair_groups = groups(pair_rows, ("table_id", "estimator"))
    ids20 = {tid for tid, game in games.items() if game["d"] == 20 and game["family"] != "overcooked"}
    require(set(pair_groups) == {(tid, arm) for tid in ids20 for arm in ("cubcr_paired_perm", "plain_perm")}, "Incomplete E4 pairing cohort")
    mse = {}
    for key, rows in pair_groups.items():
        require(len(rows) == 30 and {int(row["seed"]) for row in rows} == SEEDS, f"E4 pairing seeds: {key}")
        require(all(row["status"] == "OK" for row in rows), "Failed E4 pairing run")
        mse[key] = mean(errors(row, games[key[0]])[1] for row in rows)
    for tid in ids20:
        counts = [{int(float(row["n_total"])) for row in pair_groups[tid, arm]} for arm in ("cubcr_paired_perm", "plain_perm")]
        require(counts[0] == counts[1] and len(counts[0]) == 1, f"E4 unmatched requested budgets: {tid}")
    adaptive_rows = [row for row in csv_chunks(root, "rebuttal/results/E4/raw/adaptive") if row["eps_div"] == "20"]
    adaptive = groups(adaptive_rows, ("table_id",))
    require({key[0] for key in adaptive} == set(games), "Incomplete E4 online cohort")
    ratios = defaultdict(list)
    per_game = []
    for (tid,), rows in sorted(adaptive.items()):
        require(len(rows) == 30 and {int(row["seed"]) for row in rows} == SEEDS, f"E4 online seeds: {tid}")
        require(all(row["status"] == "eps" for row in rows), f"E4 online stopping: {tid}")
        denominator, flag = crossings[tid, "cubcr_paired_perm", 20]
        require(flag == "ok", f"E4 online denominator is censored: {tid}")
        ratio = median(float(row["n_unique"]) for row in rows) / denominator
        group = ("OC" if games[tid]["family"] == "overcooked" else "TB") + str(games[tid]["d"])
        ratios[group].append(ratio)
        per_game.append({"table_id": tid, "group": group, "online_over_reference_budget": ratio})
    return {
        "games": len(games), "grid_cells": len(cells), "grid_runs": sum(map(len, cells.values())), "seeds_per_cell": 30,
        "table4": table4, "pairing_plain_over_paired_mse": extent(mse[tid, "plain_perm"] / mse[tid, "cubcr_paired_perm"] for tid in sorted(ids20)),
        "online_over_reference_budget": {group: extent(values) for group, values in sorted(ratios.items())}, "online_per_game": per_game,
        "tb_pp_over_leverage_at_20": table4[0]["PP"]["budget"] / table4[0]["Leverage"]["budget"],
        "tb_pp_over_2add_at_100": table4[1]["PP"]["budget"] / table4[1]["2-ADD"]["budget"],
        "crossings": [{"table_id": tid, "estimator": arm, "eps_div": divisor, "budget": value, "flag": flag} for (tid, arm, divisor), (value, flag) in sorted(crossings.items())],
    }


def e5(root):
    configurations = ((1, 8), (2, 4), (2, 8), (3, 2), (3, 4))
    rows = []
    for engine, a, m, size in [("paper", a, m, size) for a, m in configurations for size in ("1e3", "1e4")] + [("acc", 4, 4, "1e3"), ("acc", 3, 8, "1e3")]:
        record = read_json(root, f"rebuttal/results/E5/raw/batch/e5__{engine}__a{a}_m{m}_S{size}_B65536.json")
        require((record["engine"], record["a"], record["m"], record["size"], record["B"], record["n"], record["k"]) == (engine, a, m, size, 65536, 16, 50), "Unexpected E5 configuration")
        require(record["status"] == "OK" and len(record["wall_s_all"]) == (5 if size == "1e3" else 3), "E5 timing repetitions")
        record["derived_wall_s"] = median(record["wall_s_all"])
        close(record["derived_wall_s"], record["wall_s"], "E5 median wall time")
        close(record["B"] * record["k"] * record["n_rows"] / record["derived_wall_s"], record["throughput"]["row_updates_per_s_wall"], "E5 throughput")
        close(2 * record["states"] * record["B"] * 8, record["analytic_memory"]["v_buffers_bytes"], "E5 value buffers")
        rows.append(record)
    submitted = [row for row in rows if row["engine"] == "paper"]
    by_config = {(row["a"], row["m"], row["size"]): row for row in submitted}
    by_size = {size: [row for row in submitted if row["size"] == size] for size in ("1e3", "1e4")}
    return {
        "submitted_batch_records": len(submitted), "separate_kernel_batch_records": len(rows) - len(submitted), "batch_size": 65536,
        "fixed_concurrency_state_time_ratio": extent(by_config[a, m, "1e4"]["derived_wall_s"] / by_config[a, m, "1e3"]["derived_wall_s"] for a, m in configurations),
        "concurrency_64_over_8_time_ratio": {size: extent(high["derived_wall_s"] / low["derived_wall_s"] for low in values for high in values if low["rows_per_state"] == 8 and high["rows_per_state"] == 64) for size, values in by_size.items()},
        "memory_gib": {size: extent(row["mem"]["delta_total_bytes"] / 2 ** 30 for row in values) for size, values in by_size.items()},
        "full_batch_table_over_value_buffers": extent(row["analytic_memory"]["table_bytes"] / row["analytic_memory"]["v_buffers_bytes"] for row in submitted),
        "separate_kernel_full_batch_s": {str(row["rows_per_state"]): row["derived_wall_s"] for row in rows if row["engine"] == "acc"},
    }


def packed(values):
    return None if values is None else b"".join(struct.pack("!d", float(value)) for value in values)


def e9(root):
    task_ids = {f"tb_none_n{n}_S{states}_k50_pf0.05_C3_s20260829" for n, states in ((16, 961), (16, 9985), (20, 961), (20, 9961), (24, 1009), (24, 9937), (28, 1009))}
    arms = {"paired_d", "paired_r", "kadd2", "leverage", "stratsvarm"}
    levels = {"B4k", "Bstop", "Bhi"}
    seeds = {20260829, 20260830, 20260831}
    expected = {(iid, level, arm, seed) for iid in task_ids for level in levels for arm in arms for seed in seeds}
    plan = read_json(root, "rebuttal/results/E9/raw/plan.json")
    plan_rows = indexed(plan["rows"], ("iid", "level", "arm"), "E9 plan")
    require(set(plan_rows) == {key[:3] for key in expected} and set(plan["seeds"]) == seeds, "Incomplete E9 plan")
    sessions = {}
    for session, directory in (("first", "gpu"), ("hostlock", "gpu_hostlock"), ("table", "table")):
        rows = [read_json(root, path.relative_to(root)) for path in sorted((root / "rebuttal/results/E9/raw" / directory).glob("*.json"))]
        data = indexed(rows, ("iid", "level", "arm", "seed"), f"E9 {session}")
        expected_session = {key for key in expected if session != "table" or "_n28_" not in key[0]}
        require(set(data) == expected_session, f"Incomplete E9 {session} run cohort")
        if session != "table":
            for key, row in data.items():
                require(row["status_run"] == "OK", f"E9 run failed: {key}")
                require(row["B"] == plan_rows[key[:3]]["B"], f"E9 run/plan budget mismatch: {key}")
                require(len(row["phi_hat"]) == row["n"] and row["k"] == 50 and row["gen_seed"] == 20260829, f"E9 run configuration: {key}")
        sessions[session] = data
    for key, first in sessions["first"].items():
        second = sessions["hostlock"][key]
        for field in ("phi_hat", "half_widths"):
            require(packed(first.get(field)) == packed(second.get(field)), f"E9 cross-session double identity: {key}/{field}")
        for field in ("n_requested", "n_engine", "engine_calls"):
            require(first[field] == second[field], f"E9 cross-session counts: {key}/{field}")
    for key, twin in sessions["table"].items():
        first = sessions["first"][key]
        require(packed(first["phi_hat"]) == packed(twin["phi_hat"]), f"E9 table-twin double identity: {key}")
        for field in ("n_requested", "n_engine", "engine_calls"):
            require(first[field] == twin[field], f"E9 table-twin counts: {key}/{field}")
    counts = {}
    cell_times = []
    for session in ("first", "hostlock"):
        medians = {key: median(row["wall_s"] for row in values) for key, values in groups(sessions[session].values(), ("iid", "level", "arm")).items()}
        for (iid, level, arm), value in sorted(medians.items()):
            cell_times.append({"session": session, "iid": iid, "level": level, "arm": arm, "wall_s": value})
        counts[session] = {
            "paired_faster_than_kadd2": sum(medians[iid, level, "paired_d"] < medians[iid, level, "kadd2"] for iid in task_ids for level in levels),
            "paired_faster_than_leverage": sum(medians[iid, level, "paired_d"] < medians[iid, level, "leverage"] for iid in task_ids for level in levels),
            "stratsvarm_faster_at_Bstop": sum(medians[iid, "Bstop", "stratsvarm"] < medians[iid, "Bstop", "paired_d"] for iid in task_ids),
            "stratsvarm_faster_at_Bhi": sum(medians[iid, "Bhi", "stratsvarm"] < medians[iid, "Bhi", "paired_d"] for iid in task_ids),
        }
    rate_groups = groups(sessions["first"].values(), ("iid", "level", "arm"))
    rates = {}
    for key, rows in rate_groups.items():
        if key[2] == "paired_r":
            continue
        require(all(len(row["resolve"]) == 2 and row["resolve"][1]["same_values"] for row in rows), f"E9 steady-state resolves: {key}")
        rates[key] = median(row["resolve"][1]["kernel_ms"] / row["resolve"][1]["B"] for row in rows)
    spread = []
    for iid in sorted(task_ids):
        for level in sorted(levels):
            values = [rates[iid, level, arm] for arm in sorted(arms - {"paired_r"})]
            spread.append(max(values) / min(values))
    return {"tasks": len(task_ids), "runs_per_session": len(expected), "bitwise_alternatives": sum(key[2] in {"kadd2", "leverage", "stratsvarm"} for key in sessions["table"]), "bitwise_all_arms": len(sessions["table"]), "cross_session_identical_runs": len(expected), "steady_state_rate_spread": extent(spread), "wall_time_counts": counts, "cell_median_wall_s": cell_times, "equal_error_wall_time_measured": False}


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate(root: Path, out: Path) -> dict:
    root, out = Path(root), Path(out)
    files = set()
    token = INPUT_FILES.set(files)
    try:
        result = {"E3": e3(root), "E4": e4(root), "E5": e5(root), "E9": e9(root)}
    finally:
        INPUT_FILES.reset(token)
    result["input_files"] = sorted(files)
    a = result["E3"]
    table3 = [{"model": "Primary grid", "cold_first_result_s": f"{a['cold_grid_total_s']['min']:.3f}-{a['cold_grid_total_s']['max']:.2f}", "model_construction_s": f"{a['cold_grid_construction_s']['min']:.3f}-{a['cold_grid_construction_s']['max']:.3f}", "policy_reattribution_s": f"{a['grid_policy_reattribution_s']['min']:.3f}-{a['grid_policy_reattribution_s']['max']:.2f}"}]
    for label, key in (("Overcooked, 20 players", "overcooked"), ("TicketBench, 300 players", "decision_time_300")):
        row = a[key]
        table3.append({"model": label, "cold_first_result_s": f"{row['cold_s']:.3g}", "model_construction_s": f"{row['construction_s']:.3g}", "policy_reattribution_s": f"{row['policy_reattribution_s']:.3g}"})
    table4 = [{"game": row["game"], "target": row["target"], **{label: row[label]["qualifier"] + str(round(row[label]["budget"])) for label in ARMS}} for row in result["E4"]["table4"]]
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "table3.csv", table3)
    write_csv(out / "table4.csv", table4)
    return result

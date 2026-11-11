import csv
import gzip
import json

import pytest

from experiments import revised_systems as systems


def write_json(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def timing_inputs(root):
    configurations = {f"n{n}_S{size}_k{k}" for n in (16, 20, 24, 28, 32)
                      for size in ("1e3", "1e4") for k in (50, 100)}
    configurations |= {"oc_k10", "dt_n12_S1e5_k25"}
    for config in configurations:
        for mode in ("cold", "warm"):
            for rep in range(6):
                phase = dict(config=config, mode=mode, rep=rep, interp_start_s=.1,
                             imports_s=.2, build_p0_s=1., build_p0m_s=2.,
                             build_p1_s=3., marshal_s=4., p2_s=5.,
                             p3a_context_s=6., p3b_alloc_s=7., p4_upload_s=8.,
                             ups_solve_s=9., p5_s=10., p6_assembly_s=11., total_s=66.3)
                if rep == 0:
                    phase["total_s"] = 1000.
                write_json(root, f"rebuttal/results/E3/raw/phases/{config}__{mode}__{rep}.json", phase)
        revisions = ["R1"]
        if config in {f"n{n}_S{size}_k50" for n in (16, 28, 32) for size in ("1e3", "1e4")}:
            revisions += ["R2", "R3"]
        if config == "oc_k10":
            revisions += ["R3"]
        for revision in revisions:
            for rep in range(1, 6):
                row = dict(config=config, revision=revision, rep=rep, update_s=.1,
                           ups_solve_s=.2, p5_s=.3, reattr_s=.6, p1_reassembly_s=.4,
                           reattr_incl_p1_s=1., fresh_create_submitted_s=.15,
                           correctness=dict(endpoints_bitwise=True, random_bitwise=True,
                                            sampler_same_as_fresh=None if config.startswith("dt") and rep > 1 else True),
                           revert_endpoints_bitwise=True)
                write_json(root, f"rebuttal/results/E3/raw/revisions/{config}__{revision}__{rep}.json", row)
    write_json(root, "rebuttal/results/E3/raw/revisions/oc_k10__R2__NA.json",
               {"status": "NOT APPLICABLE"})


def test_latency_uses_raw_records_excluding_warmups(tmp_path):
    timing_inputs(tmp_path)
    result = systems.e3(tmp_path)
    assert result["phase_runs"] == 220
    assert result["revision_runs"] == 175
    assert result["complete_sampler_matches"] == 171
    assert result["cold_grid_total_s"]["median"] == pytest.approx(66.3)
    assert result["cold_grid_construction_s"]["median"] == pytest.approx(15.)
    assert result["overcooked"]["policy_reattribution_s"] == pytest.approx(1.)


def test_latency_rejects_missing_repetition(tmp_path):
    timing_inputs(tmp_path)
    (tmp_path / "rebuttal/results/E3/raw/phases/oc_k10__cold__1.json").unlink()
    with pytest.raises(ValueError, match="E3 repetitions"):
        systems.e3(tmp_path)


def test_latency_rejects_inconsistent_phase_total(tmp_path):
    timing_inputs(tmp_path)
    path = tmp_path / "rebuttal/results/E3/raw/phases/oc_k10__cold__1.json"
    value = json.loads(path.read_text())
    value["total_s"] += 1.
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="complete first-result total"):
        systems.e3(tmp_path)


def test_estimator_errors_use_generated_game_reference():
    game = {"d": 2, "phi_exact": [.25, .75]}
    row = {"table_id": "synthetic", "phi_hat": ".375 .625", "max_abs": .125,
           "mse": .015625}
    assert systems.errors(row, game) == (.125, .015625)
    with pytest.raises(ValueError, match="Exact attribution vector"):
        systems.errors(dict(row, phi_exact=".5 .5"), game)


def test_raw_chunks_track_provenance_without_merged_results(tmp_path):
    relative = "rebuttal/results/E4/raw/grid/part.csv.gz"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["table_id", "seed"])
        writer.writeheader()
        writer.writerow({"table_id": "synthetic", "seed": 1})
    files = set()
    token = systems.INPUT_FILES.set(files)
    try:
        rows = list(systems.csv_chunks(tmp_path, "rebuttal/results/E4/raw/grid"))
    finally:
        systems.INPUT_FILES.reset(token)
    assert rows == [{"table_id": "synthetic", "seed": "1"}]
    assert files == {relative}

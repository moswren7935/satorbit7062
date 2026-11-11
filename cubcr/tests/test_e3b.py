import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from cubcr import io_prism, vi_ref
from experiments import e3b_bench as b
from experiments.e3_prism import parse_prism_masks

TOL = 1e-6


def test_benchmark_configuration():
    assert len(b.BENCH3B) == 9
    ids = [e["id"] for e in b.BENCH3B]
    assert len(set(ids)) == len(ids), "duplicate benchmark ids"
    for e in b.BENCH3B:
        for key in ("id", "path", "source", "paper", "players", "consts", "k", "fail"):
            assert key in e, f"{e.get('id')}: missing {key}"
        assert e["source"] in ("qest18", "fm19nash")
        assert e["paper"] in ("KNPS18", "KNPS19")
        assert isinstance(e["players"], int) and 1 <= e["players"] <= 4
        assert isinstance(e["k"], int) and e["k"] > 0
        assert isinstance(e["consts"], dict)
        fkw = e["fail"]
        assert set(fkw) <= {"fail_label", "fail_expr"} and len(fkw) == 1, fkw

def test_bench3b_paths_inside_dataset():
    for e in b.BENCH3B:
        assert not os.path.isabs(e["path"])
        assert os.path.realpath(b.resolve_path(e)).startswith(os.path.realpath(b.DATASET))


def test_feasibility_recorded_for_kept_models():
    for e in b.BENCH3B:
        f = b.FEASIBILITY.get(e["id"])
        assert f is not None, f"{e['id']}: no recorded feasibility probe"
        assert f["status"] == "OK"
        assert f["n"] == e["players"] <= 4
        assert 0 < f["n_states"] <= 50000


        assert 0.0 < f["v_allpinned"] <= 1.0
        assert 0.0 <= f["v_allfree"] <= f["v_allpinned"] + 1e-12


def test_reference_fixtures_match_benchmark_ids():
    kept = {e["id"] for e in b.BENCH3B}
    assert set(b.FEASIBILITY) == kept
    assert all(f["status"] == "OK" for f in b.FEASIBILITY.values())


def test_active_bench_shape():
    ab = b.active_bench()
    assert len(ab) == len(b.BENCH3B)
    iids = [t[0] for t in ab]
    assert len(set(iids)) == len(iids)
    for iid, path, consts, k, fkw, entry in ab:
        assert iid.startswith("e3b_")
        assert os.path.isfile(path)
        assert isinstance(consts, dict) and isinstance(k, int)
        assert set(fkw) <= {"fail_label", "fail_expr"} and len(fkw) == 1


def test_parse_prism_masks():
    assert parse_prism_masks("", 3) is None
    assert parse_prism_masks("extremes", 2) == {0, 3}
    assert parse_prism_masks("0,5", 3) == {0, 5}
    with pytest.raises(ValueError):
        parse_prism_masks("8", 3)


def test_benchmark_driver_records_operational_metadata(tmp_path, monkeypatch):
    from experiments import e3_prism

    monkeypatch.setattr(e3_prism.protocol, "machine_tags", lambda: {})
    monkeypatch.setattr(e3_prism.io_prism, "tool_identity", lambda: {})
    imported = []
    records = []

    def import_model(path, consts, **kwargs):
        imported.append((path, consts, kwargs))
        return object()

    def run_instance(*args, extra, **kwargs):
        records.append(extra)
        return {"gate_failed": None}

    monkeypatch.setattr(e3_prism.io_prism, "import_prism_csg", import_model)
    monkeypatch.setattr(e3_prism, "run_instance", run_instance)
    assert e3_prism.main(["--bench-file", "e3b", "--only", "ids_scenario1",
                          "--out", str(tmp_path / "out"), "--work", str(tmp_path / "work")]) == 0
    assert len(imported) == len(records) == 1
    assert imported[0][1] == {"rounds": 25, "K": 1}
    assert imported[0][2]["k"] == 25 and imported[0][2]["fail_expr"] == "s=3"
    assert set(records[0]) == {"consts", "fail", "bench_path", "bench_source", "bench_paper"}
    assert records[0]["bench_source"] == "qest18" and records[0]["bench_paper"] == "KNPS18"


@pytest.mark.skipif(not io_prism.prism_available(),
                    reason=f"PRISM-games binary not found at {io_prism.PRISM_BIN}")
def test_import_roundtrip_cheapest_model(tmp_path):


    entry = next(e for e in b.BENCH3B if e["id"] == "ids_scenario1")
    f = b.FEASIBILITY[entry["id"]]
    csg = io_prism.import_prism_csg(b.resolve_path(entry), entry["consts"], policy="uniform",
                                    k=entry["k"], out_base=str(tmp_path / "imp"), timeout_s=60.0,
                                    **entry["fail"])
    assert csg.n_agents == f["n"] and csg.n_states == f["n_states"]
    n, k = csg.n_agents, entry["k"]
    for mask in (0, (1 << n) - 1):
        ref = float(vi_ref.solve(csg, mask, k))
        model, props = io_prism.emit_coalition_model(csg, mask, k, comment="e3b roundtrip test")
        r = io_prism.run_prism(model, props, timeout_s=60.0, workdir=str(tmp_path),
                               tag=f"rt_mask{mask}")
        assert r["status"] == "OK", r
        assert abs(r["value"] - ref) <= TOL, (mask, r["value"], ref)

    assert abs(float(vi_ref.solve(csg, 0, k)) - f["v_allfree"]) <= 1e-9
    assert abs(float(vi_ref.solve(csg, (1 << n) - 1, k)) - f["v_allpinned"]) <= 1e-9

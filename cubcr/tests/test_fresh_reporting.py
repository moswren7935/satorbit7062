import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from experiments import e9v3_certified, revised_paper


@pytest.mark.parametrize('child_code', [
    "import sys; print('failed', file=sys.stderr); sys.exit(1)",
    "print('incomplete')",
])
def test_genuine_exact_failure_is_recorded_and_raised(tmp_path, monkeypatch, child_code):
    real_popen = subprocess.Popen
    monkeypatch.setattr(subprocess, 'Popen', lambda cmd, **kw: real_popen([sys.executable, '-c', child_code], **kw))
    monkeypatch.setattr(e9v3_certified, 'RESULTS', str(tmp_path))
    monkeypatch.setattr(e9v3_certified, 'REPO', str(tmp_path))
    monkeypatch.setattr(e9v3_certified, 'lib_hashes', lambda: {})
    monkeypatch.setattr(e9v3_certified, 'build_model', lambda *args: (SimpleNamespace(n_states=10, nnz=20), 'tiny', {}, 0))
    with pytest.raises(RuntimeError, match='exact_cpu'):
        e9v3_certified.genuine_to(dict(n=2, states=10, k=1, pattern='none'), 20260829, 5, {})
    record = json.loads((tmp_path / 'e9v3__tiny__exact_cpu_genuineTO.json').read_text())
    assert record['status'] == 'CRASH'
    assert record['wall_s'] is None
    assert record['censored'] is False
    assert record['child_stdout'] or record['child_stderr']


def test_genuine_exact_success_requires_completed_child(tmp_path, monkeypatch):
    real_popen = subprocess.Popen
    monkeypatch.setattr(subprocess, 'Popen', lambda cmd, **kw: real_popen([sys.executable, '-c', "print('DONE 0.5')"], **kw))
    monkeypatch.setattr(e9v3_certified, 'RESULTS', str(tmp_path))
    monkeypatch.setattr(e9v3_certified, 'REPO', str(tmp_path))
    monkeypatch.setattr(e9v3_certified, 'lib_hashes', lambda: {})
    monkeypatch.setattr(e9v3_certified, 'build_model', lambda *args: (SimpleNamespace(n_states=10, nnz=20), 'tiny', {}, 0))
    e9v3_certified.genuine_to(dict(n=2, states=10, k=1, pattern='none'), 20260829, 5, {})
    record = json.loads((tmp_path / 'e9v3__tiny__exact_cpu_genuineTO.json').read_text())
    assert record['status'] == 'OK'
    assert record['wall_s'] > 0


def test_genuine_exact_timeout_stays_censored(tmp_path, monkeypatch):
    real_popen = subprocess.Popen
    monkeypatch.setattr(subprocess, 'Popen', lambda cmd, **kw: real_popen([sys.executable, '-c', 'import time; time.sleep(10)'], **kw))
    monkeypatch.setattr(e9v3_certified, 'RESULTS', str(tmp_path))
    monkeypatch.setattr(e9v3_certified, 'REPO', str(tmp_path))
    monkeypatch.setattr(e9v3_certified, 'lib_hashes', lambda: {})
    monkeypatch.setattr(e9v3_certified, 'build_model', lambda *args: (SimpleNamespace(n_states=10, nnz=20), 'tiny', {}, 0))
    e9v3_certified.genuine_to(dict(n=2, states=10, k=1, pattern='none'), 20260829, .05, {})
    record = json.loads((tmp_path / 'e9v3__tiny__exact_cpu_genuineTO.json').read_text())
    assert record['status'] == 'TO'
    assert record['wall_s'] >= .05
    assert record['censored'] is True


@pytest.fixture
def report_inputs(tmp_path):
    def put(path, value):
        path = tmp_path / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    for n in (16, 20, 24, 28, 32):
        for states in (1000, 10000):
            for k in (50, 100):
                for gs in (20260829, 20260830, 20260831):
                    iid = f'n{n}_S{states}_k{k}_g{gs}'
                    base = dict(instance_id=iid, n=n, states=states, k=k, gen_seed=gs, cell=dict(states=states),
                                status='OK', censored=False, n_solves=10, n_perms=4, n_units=2,
                                batch_sizes=[2], shapley_sampled=[.5] * n)
                    put(f'results/e9v3/e9v3__{iid}__exact_cpu.json', dict(base, wall_s=20))
                    for seed in (20260829, 20260830, 20260831):
                        for engine, wall in (('gpu', 2), ('cpu', 6)):
                            put(f'results/e9v3/e9v3__{iid}__cert_{engine}_s{seed}.json', dict(base, seed=seed, wall_s=wall))
    for n in (8, 12, 16, 20, 24):
        for states in (1000, 10000, 100000, 1000000):
            for k in (25, 100, 400):
                for seed in (20260829, 20260830, 20260831):
                    iid = f'n{n}_S{states}_k{k}_s{seed}'
                    for engine, threads, wall in (('gpu_batched', 0, 2), ('cpu_multicore', 32, 8)):
                        put(f'results/e1/{iid}__{engine}.json', dict(instance_id=iid, cell_id=iid, n=n, states=states,
                            k=k, seed=seed, status='OK', censored=False, engine=engine, threads=threads,
                            wall_s=wall, mask_sha256='same', n_coalitions=256, dtype='fp64'))
    for n in (8, 12):
        for states in (1000, 10000):
            for k in (25, 100):
                iid = f'n{n}_S{states}_k{k}'
                for engine, wall in (('gpu_batched', 2), ('gpu_unbatched', 40)):
                    put(f'results/e2/{iid}__{engine}.json', dict(instance_id=iid, n=n, states=states, k=k,
                        status='OK', censored=False, engine=engine, wall_s=wall, mask_sha256='same',
                        n_coalitions=2**n, dtype='fp64', full_enumeration=True))
    return tmp_path


def test_primary_report_uses_available_exact_references(report_inputs):
    result = revised_paper.summarize_primary(report_inputs, report_inputs / 'report')
    assert result['tasks_with_cpu_reference'] == 60
    assert result['combined_ratios'] == 180
    assert result['combined_speedup'] == 10
    assert result['cpu_gpu_range'] == [3, 3]


def test_report_accepts_censored_runs_without_treating_caps_as_measurements(report_inputs):
    for path in (report_inputs / 'results/e9v3').glob('*__exact_cpu.json'):
        data = json.loads(path.read_text())
        data.update(status='NOT RUN', wall_s=None)
        path.write_text(json.dumps(data))
    for family, pattern in [('e9v3', '*__cert_cpu_s20260829.json'), ('e1', '*__cpu_multicore.json'), ('e2', '*__gpu_unbatched.json')]:
        for path in (report_inputs / 'results' / family).glob(pattern):
            data = json.loads(path.read_text())
            data.update(status='TO', censored=True, wall_s=900)
            path.write_text(json.dumps(data))
    result = revised_paper.summarize_primary(report_inputs, report_inputs / 'report')
    assert result['combined_speedup'] is None
    assert result['tasks_with_cpu_reference'] == 0
    assert result['cpu_gpu_range'] == [3, 3]
    assert result['primary_matched_completed_runs'] == 120
    assert result['identical_batch_speedup'] is None
    assert result['batching_speedup_range'] == [None, None]
    revised_paper.figures(report_inputs / 'report', result)
    assert (report_inputs / 'report/figure4.pdf').is_file()


def test_report_still_rejects_missing_planned_engine_run(report_inputs):
    next((report_inputs / 'results/e1').glob('*__cpu_multicore.json')).unlink()
    with pytest.raises(ValueError, match='identical-batch'):
        revised_paper.summarize_primary(report_inputs, report_inputs / 'report')


def test_report_with_no_completed_runs_keeps_empty_measurements(report_inputs):
    for path in (report_inputs / 'results').rglob('*.json'):
        data = json.loads(path.read_text())
        data.update(status='TO', censored=True, wall_s=900)
        path.write_text(json.dumps(data))
    result = revised_paper.summarize_primary(report_inputs, report_inputs / 'report')
    assert result['combined_speedup'] is None
    assert result['cpu_gpu_range'] == [None, None]
    assert result['primary_matched_completed_runs'] == 0
    assert result['identical_batch_speedup'] is None
    assert result['batching_speedup_range'] == [None, None]
    revised_paper.figures(report_inputs / 'report', result)
    json.dumps(result, allow_nan=False)
    assert (report_inputs / 'report/figure3.pdf').is_file()

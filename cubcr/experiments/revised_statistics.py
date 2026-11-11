import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

PAPER_SEEDS = (20260829, 20260830, 20260831)
NEW_SEEDS = tuple(range(20261101, 20261121))
DIVISORS = (10, 20, 50, 100)
PRIMARY_STATES = {16: (961, 9985), 20: (961, 9961), 24: (1009, 9937), 28: (1009, 10081), 32: (961, 9985)}
DECISION_MODELS = ((3, 55, 4), (8, 1009, 25), (8, 9985, 25), (12, 100009, 25), (16, 961, 25), (16, 9985, 25))


class _Inputs:
    def __init__(self, root):
        self.root = Path(root)
        self.paths = set()

    def require(self, relative):
        path = self.root / relative
        if not path.is_file():
            raise FileNotFoundError(f'Missing required input: {relative}')
        self.paths.add(path.relative_to(self.root).as_posix())
        return path

    def json(self, relative):
        path = self.require(relative)
        with path.open(encoding='utf-8') as handle:
            value = json.load(handle)
        value['_source'] = path.relative_to(self.root).as_posix()
        return value

    def records(self, relative, pattern, select=lambda value: True, allow_empty=False):
        result = []
        for path in sorted((self.root / relative).glob(pattern)):
            with path.open(encoding='utf-8') as handle:
                value = json.load(handle)
            if select(value):
                self.require(path.relative_to(self.root))
                value['_source'] = path.relative_to(self.root).as_posix()
                result.append(value)
        if not result and not allow_empty:
            raise ValueError(f'Missing required cohort: {relative}/{pattern}')
        return result

    def csv(self, relative):
        return pd.read_csv(self.require(relative))


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _keys(records, fields, label, expected=None):
    found = {}
    for row in records:
        key = tuple(row[field] for field in fields)
        if key in found:
            raise ValueError(f'Duplicate {label} record {key}: {found[key].get("_source", "CSV row")} and {row.get("_source", "CSV row")}')
        found[key] = row
    if expected is not None:
        wanted = set(expected)
        missing, extra = wanted - set(found), set(found) - wanted
        _check(not missing and not extra, f'Incomplete {label} cohort: missing {len(missing)}, unexpected {len(extra)}; examples {sorted(map(str, missing))[:3]} {sorted(map(str, extra))[:3]}')
    return found


def _count(rows, expected, label):
    _check(len(rows) == expected, f'Incomplete {label} cohort: expected {expected} rows, found {len(rows)}')


def _med(values):
    values = np.asarray(values, dtype=float)
    _check(values.size > 0 and np.isfinite(values).all(), 'Median requires a nonempty finite cohort')
    return float(np.median(values))


def _array(value):
    result = np.asarray(value, dtype=float)
    _check(result.ndim == 1 and np.isfinite(result).all(), 'Expected a finite one-dimensional score or interval array')
    return result


def _ranking(estimate, lower, upper, reference=None):
    est, lo, hi = map(_array, (estimate, lower, upper))
    _check(len(est) > 3 and est.shape == lo.shape == hi.shape and np.all(lo <= hi), 'Invalid ranking arrays')
    chosen = np.argsort(-est, kind='stable')
    result = {'sep1': bool(lo[chosen[0]] > hi[chosen[1:]].max()), 'sep3': bool(lo[chosen[:3]].min() > hi[chosen[3:]].max()), 'pos': int((lo > 0).sum())}
    if reference is not None:
        phi = _array(reference)
        _check(phi.shape == est.shape, 'Reference and estimate dimensions differ')
        exact = np.argsort(-phi, kind='stable')
        result.update(top1=bool(phi.max() - phi[chosen[0]] <= 1e-12), top3=set(exact[:3]) == set(chosen[:3]), rank=int(np.where(exact == chosen[0])[0][0] + 1), cover=bool(np.all((lo <= phi) & (phi <= hi))), error=float(abs(est - phi).max()), shortfall=float(phi.max() - phi[chosen[0]]), exact=exact, chosen=chosen)
    return result


def _primary_ids():
    return {f'tb_none_n{n}_S{states}_k{k}_pf0.05_C3_s{seed}' for n, states_pair in PRIMARY_STATES.items() for states, k, seed in itertools.product(states_pair, (50, 100), PAPER_SEEDS)}


def _decision_ids():
    return [f'tb_none_n{n}_S{states}_k{k}_pf0.05_C3_s20260829' for n, states, k in DECISION_MODELS]


def _exact_references(source):
    exact = {}
    for iid, arm in itertools.product(sorted(_primary_ids()), ('cpu', 'gpu')):
        record = source.json(f'results/e9v3/e9v3__{iid}__exact_{arm}.json')
        _check(record['instance_id'] == iid, f'Incorrect exhaustive reference identity: {record["_source"]}')
        if record.get('status') == 'OK' and not record.get('censored'):
            _check(record['wall_s'] > 0 and record.get('shapley') is not None, 'Completed exhaustive run needs positive runtime and attribution values')
            exact[iid, arm] = record
    return exact


def _optional_stat(values, operation='median'):
    values = pd.Series(values, dtype=float).dropna()
    return None if values.empty else float(getattr(values, operation)())


def _timing(source):
    primary_ids = _primary_ids()
    exact = _exact_references(source)
    reference_ids = {iid for iid, arm in exact if arm == 'gpu'}
    all_reference_ids = {iid for iid, arm in exact}
    primary = []
    originals = {}
    for iid, seed in itertools.product(sorted(primary_ids), PAPER_SEEDS):
        gpu = source.json(f'results/e9v3/e9v3__{iid}__cert_gpu_s{seed}.json')
        cpu = source.json(f'results/e9v3/e9v3__{iid}__cert_cpu_s{seed}.json')
        for row in (gpu, cpu):
            _check(row['instance_id'] == iid and row['seed'] == seed, f'Incorrect original run identity: {row["_source"]}')
            _check(row['wall_s'] > 0, 'Sampled runtime must be positive')
        originals[iid, seed] = gpu
        primary.append({'iid': iid, 'seed': seed, 'n': gpu['n'], 'S': 1000 if gpu['states'] < 2000 else 10000, 'k': gpu['k'], 'ratio': cpu['wall_s'] / gpu['wall_s'], 'cpu_wall': cpu['wall_s'], 'gpu_wall': gpu['wall_s'], 'gpu_eps': gpu['sampling_status'] == 'eps', 'cpu_eps': cpu['sampling_status'] == 'eps', 'same_batch_sizes': gpu['batch_sizes'] == cpu['batch_sizes'], 'same_estimates': gpu['shapley_sampled'] == cpu['shapley_sampled'], 'same_solves': gpu['n_solves'] == cpu['n_solves']})
    pf = pd.DataFrame(primary)
    config = pf.groupby(['n', 'S', 'k']).ratio.median()
    by_task = pf.groupby(['n', 'S', 'k', 'iid'])[['cpu_wall', 'gpu_wall']].median()
    nested = (by_task.cpu_wall / by_task.gpu_wall).groupby(['n', 'S', 'k']).median()
    paired = {'matched_runs': len(pf), **{name: int(pf[name].sum()) for name in ('gpu_eps', 'cpu_eps', 'same_batch_sizes', 'same_estimates', 'same_solves')}, 'configuration_median_ratio_range': [float(config.min()), float(config.max())], 'configuration_nested_ratio_range': [float(nested.min()), float(nested.max())]}
    records = source.records('rebuttal/results/E2/raw/partB_grid', 'partB_grid__*.json', lambda d: d['eps_div'] in DIVISORS)
    _keys(records, ('instance_id', 'eps_div', 'seed'), 'E2 GPU timing', itertools.product(primary_ids, DIVISORS, PAPER_SEEDS))
    rows = []
    for d in records:
        iid, seed, wall = d['instance_id'], d['seed'], d['wall_s']
        _check(wall > 0 and d['status'] in ('eps', 'time', 'max_perms'), f'Invalid tolerance run: {d["_source"]}')
        _check(bool(d['certified']) == (d['status'] == 'eps'), f'Inconsistent tolerance status: {d["_source"]}')
        row = {'iid': iid, 'div': d['eps_div'], 'seed': seed, 'wall_s': wall, 'certified': bool(d['certified']), 'status': d['status'], 'n': d['n'], 'states': d['states'], 'k': d['k'], 'original_wall_s': originals[iid, seed]['wall_s'], 'err': None, 'cpu': None, 'gpu': None, 'original_cpu': None, 'original_gpu': None}
        _check((d.get('phi_exact') is not None) == (iid in all_reference_ids), f'Missing or unexpected reference: {d["_source"]}')
        if iid in all_reference_ids:
            row['err'] = float(np.max(abs(_array(d['phi_hat']) - _array(d['phi_exact']))) / d['upsilon_today'])
        for arm in ('cpu', 'gpu'):
            if (iid, arm) in exact:
                if d['certified']:
                    row[arm] = exact[iid, arm]['wall_s'] / wall
                if originals[iid, seed]['sampling_status'] == 'eps':
                    row['original_' + arm] = exact[iid, arm]['wall_s'] / row['original_wall_s']
        rows.append(row)
    frame = pd.DataFrame(rows)
    table = {}
    for div, g in frame.groupby('div'):
        table[str(div)] = {'runs': len(g), 'tasks': int(g.iid.nunique()), 'certified_runs': int(g.certified.sum()), 'status_counts': {str(status): int(count) for status, count in g.status.value_counts().items()}, 'median_time': _med(g.wall_s), 'errors': int(g.err.notna().sum()), 'median_error_pct': _optional_stat(g.err * 100), 'max_error_pct': _optional_stat(g.err * 100, 'max'), 'cpu_ratios': int(g.cpu.notna().sum()), 'cpu_median_ratio': _optional_stat(g.cpu), 'gpu_ratios': int(g.gpu.notna().sum()), 'gpu_median_ratio': _optional_stat(g.gpu), 'gpu_slower': int((g.gpu < 1).sum())}
        _check(table[str(div)]['errors'] == len(all_reference_ids) * len(PAPER_SEEDS), f'Incomplete E2 matched-reference cohort at divisor {div}')
        for arm in ('cpu', 'gpu'):
            expected = sum(row['certified'] and (row['iid'], arm) in exact for row in g.to_dict('records'))
            _check(table[str(div)][arm + '_ratios'] == expected, f'Incomplete {arm} completed-speedup cohort at divisor {div}')
    g = frame[frame['div'] == 20]
    original = {'cpu_ratios': int(g.original_cpu.notna().sum()), 'cpu_median_ratio': _optional_stat(g.original_cpu), 'gpu_ratios': int(g.original_gpu.notna().sum()), 'gpu_median_ratio': _optional_stat(g.original_gpu), 'small_state_range': [float(g[g.states < 2000].original_wall_s.min()), float(g[g.states < 2000].original_wall_s.max())], 'large_state_range': [float(g[g.states > 2000].original_wall_s.min()), float(g[g.states > 2000].original_wall_s.max())]}
    slower = frame[(frame['div'] == 100) & (frame.gpu < 1)].groupby(['n', 'states', 'k']).size()
    return {'E2B_table2': table, 'original_runtime': original, 'E2B_slower_cells': {str(k): int(v) for k, v in slower.items()}, 'primary_cpu_gpu': paired}, reference_ids


def _accuracy(source, reference_ids):
    records = source.records('rebuttal/results/E2/raw/partA_runs', '*.json', lambda d: d['div'] in (20, 50, 100))
    expected = {('ticketbench', iid, div, seed) for iid, div, seed in itertools.product(reference_ids, (20, 50, 100), PAPER_SEEDS + NEW_SEEDS)}
    expected |= {('overcooked', 'overcooked_cramped_room_k10', div, seed) for div, seed in itertools.product((20, 50, 100), PAPER_SEEDS + NEW_SEEDS)}
    _keys(records, ('family', 'iid', 'div', 'sampler_seed'), 'E2 accuracy', expected)
    rows = []
    for d in records:
        r = d['result']
        rank = _ranking(r['phi_hat'], r['L'], r['R'], d['phi_exact'])
        phi, h = _array(d['phi_exact']), _array(r['half_widths'])
        exact = rank.pop('exact')
        rank.pop('chosen')
        _check(d['upsilon'] > 0 and d['eps20'] > 0, 'Accuracy normalizers must be positive')
        rows.append({**rank, 'family': d['family'], 'iid': d['iid'], 'div': d['div'], 'seed': d['sampler_seed'], 'intervals': len(phi), 'err': rank['error'] / d['upsilon'], 'shortfall': rank['shortfall'] / d['upsilon'], 'cover_h': bool(np.all(abs(_array(r['phi_hat']) - phi) <= h)), 'gap': float((phi[exact[0]] - phi[exact[1]]) / d['eps20']), 'top3mass': float(phi[exact[:3]].sum() / d['upsilon']), **{f'half_rel_{i+1}': float(h[exact[i]] / phi[exact[i]]) for i in range(3)}})
    frame = pd.DataFrame(rows)
    new = {}
    for (family, div), g in frame[frame.seed.isin(NEW_SEEDS)].groupby(['family', 'div']):
        new[f'{family}_{div}'] = {'runs': len(g), 'tasks': int(g.iid.nunique()), **{name: int(g[name].sum()) for name in ('top1', 'top3', 'sep1', 'sep3')}, 'max_error_pct': float(g.err.max() * 100), 'max_shortfall_pct': float(g.shortfall.max() * 100), 'wrong_selected_ranks': sorted(g[~g.top1]['rank'].unique().tolist()), 'tasks_below_observed_95pct': int((g.groupby('iid').top1.mean() < .95).sum())}
    g = frame[(frame.family == 'ticketbench') & frame.seed.isin(PAPER_SEEDS) & (frame['div'] == 20)]
    paper = {'runs': len(g), 'median_error_pct': _optional_stat(g.err * 100), 'max_error_pct': _optional_stat(g.err * 100, 'max'), 'top3': int(g.top3.sum()), 'top3_mass_median': _optional_stat(g.top3mass), 'relative_halfwidth_medians': [_optional_stat(g[f'half_rel_{k}']) for k in (1, 2, 3)], 'leading_gap_over_eps_range': [_optional_stat(g.gap, 'min'), _optional_stat(g.gap, 'max')]}
    return {'E2A_overall': {'runs': len(frame), 'covered': int(frame.cover.sum()), 'covered_h': int(frame.cover_h.sum()), 'intervals': int(frame.intervals.sum()), 'divisors': [20, 50, 100]}, 'E2A_new': new, 'E2A_paper20': paper}


def _stopping(source, reference_ids):
    records = source.records('rebuttal/results/E7/raw/gpu', '*.json', lambda d: d.get('K') in (1, 3), allow_empty=not reference_ids)
    _keys(records, ('iid', 'K', 'sampler_seed'), 'E7 ranking stop', itertools.product(reference_ids, (1, 3), PAPER_SEEDS))
    rows = []
    for d in records:
        k = d['K']
        rank = _ranking(d['phi_hat'], d['L'], d['R'], d['phi_exact'])
        separated = rank[f'sep{k}']
        _check(separated == d['certified'], f'Stored separation flag disagrees with bounds: {d["_source"]}')
        rows.append({'K': k, 'sep': separated, 'set_correct': set(rank['chosen'][:k]) == set(rank['exact'][:k]), 'covered': rank['cover'], 'wall': d['wall_s'], 'status': d['status'], 'iid': d['iid'], 'seed': d['sampler_seed'], 'max_perms': d['config']['max_perms'], 'time_budget': d['config']['time_budget']})
    result = {}
    for k, g in pd.DataFrame(rows).groupby('K') if rows else []:
        result[str(k)] = {'runs': len(g), 'separated': int(g.sep.sum()), 'correct_separated': int(g[g.sep].set_correct.sum()), 'correct_all_endpoint_estimates': int(g.set_correct.sum()), 'covered': int(g.covered.sum()), 'median_wall_all': _med(g.wall), 'median_wall_separated': _optional_stat(g[g.sep].wall), 'censored': g[~g.sep][['iid', 'seed', 'status', 'wall']].to_dict('records'), 'permutation_caps': sorted(g.max_perms.unique().tolist()), 'time_caps_s': sorted(g.time_budget.unique().tolist())}
    return result


def _decision(source):
    ids = _decision_ids()
    records = source.records('results/e4v2', '*__eps*__s*.json')
    expected = itertools.product(ids, (.005, .01, .02), PAPER_SEEDS)
    keyed = [{**d, 'eps': d['rule_params']['eps']} for d in records]
    _keys(keyed, ('instance_id', 'eps', 'seed'), 'original decision-time', expected)
    rows = []
    for d in records:
        cert = d['certificate']
        rank = (_ranking(cert['estimates'], _array(cert['cs_lower']) - cert['fp_term'], _array(cert['cs_upper']) + cert['fp_term'])
                if cert else {'pos': None, 'sep1': None, 'sep3': None})
        rows.append({'iid': d['instance_id'], 'eps': d['rule_params']['eps'], 'n': d['n'], 'states': d['states'], 'k': d['k'], 'd': d['n_players'], 'certified': d['certified'], 't': d['time_to_certify_s'], 'queries': d['sampling']['n_solves'], 'pos_lower': rank['pos'], 'sep1': rank['sep1'], 'sep3': rank['sep3'], 'fp': d['sampler_v2']['fp_term'], 'reachable': d['instance_meta']['n_reachable_k']})
    frame = pd.DataFrame(rows)
    table = frame[frame.eps == .005].groupby('iid').agg(n=('n', 'first'), states=('states', 'first'), k=('k', 'first'), players=('d', 'first'), certified_runs=('certified', 'sum'), queries=('queries', 'median'), time=('t', 'median'), positive_lower_min=('pos_lower', 'min'), positive_lower_max=('pos_lower', 'max'), reachable=('reachable', 'first'), sep1=('sep1', 'sum'), sep3=('sep3', 'sum')).to_dict('index')
    original = {'runs': len(frame), 'certified': int(frame.certified.sum()), 'eps': sorted(frame.eps.unique().tolist()), 'beta_range': [float(frame.fp.min()), float(frame.fp.max())], 'table1': table}
    records = source.records('rebuttal/results/E2/raw/partB_dt', 'partB_dt__*.json', lambda d: d['eps_abs'] == .001)
    _keys(records, ('instance_id', 'seed'), 'tight decision-time', itertools.product(ids, PAPER_SEEDS))
    rows = []
    for d in records:
        rank = _ranking(d['phi_hat'], d['L'], d['R'])
        rows.append({'iid': d['instance_id'], 'd': d['n_players'], 't': d['time_to_certify_s'], 'pos': rank['pos'], 'sep1': rank['sep1'], 'sep3': rank['sep3']})
    tight = pd.DataFrame(rows).groupby('iid').agg(runs=('iid', 'size'), d=('d', 'first'), time=('t', 'median'), positive_lower_median=('pos', 'median'), sep1=('sep1', 'sum'), sep3=('sep3', 'sum')).to_dict('index')
    records = source.records('rebuttal/results/E2/raw/partB_oc', 'partB_oc__*.json', lambda d: d['eps_div'] == 100)
    _keys(records, ('instance_id', 'seed'), 'Overcooked tight timing', itertools.product(('overcooked_cramped_room_k10',), PAPER_SEEDS))
    return {'original_dt': original, 'dt_001': tight, 'oc_100_time': {'runs': len(records), 'median': _med([d['wall_s'] for d in records])}}


def _loo(source):
    records = source.records('results/e12', 'e12__ticketbench_new__*.json')
    _keys(records, ('pattern', 'n', 'k', 'seed'), 'controlled LOO', itertools.product(('none', 'dummy', 'symmetric', 'joint', 'redundant'), (8, 12, 16), (25, 100), PAPER_SEEDS))
    _keys(records, ('instance_id',), 'controlled LOO model')
    rows = []
    for d in records:
        a, phi = _array(d['loo_remove']), _array(d['shapley'])
        _check(a.shape == phi.shape, 'LOO and Shapley dimensions differ')
        i, j = np.triu_indices(len(a), 1)
        tau = np.mean(np.sign(a[i] - a[j]) * np.sign(phi[i] - phi[j]))
        oa, op = np.argsort(-a, kind='stable'), np.argsort(-phi, kind='stable')
        rows.append({'pattern': d['pattern'], 'tau': tau, 'top1': oa[0] == op[0], 'max_disp': int(abs(np.argsort(oa) - np.argsort(op)).max())})
    return {pattern: {'runs': len(g), 'median_tau_a': _med(g.tau), 'top1': int(g.top1.sum()), 'max_disp': int(g.max_disp.max())} for pattern, g in pd.DataFrame(rows).groupby('pattern')}


def _interval_audit(source, reference_ids):
    records = source.records('rebuttal/results/E1/raw/interval_runs', '*.json')
    _keys([d['row'] for d in records], ('run_id',), 'E1 replay run')
    _count(records, len(_primary_ids()) * len(PAPER_SEEDS) + 13 * 8 * len(PAPER_SEEDS) + len(PAPER_SEEDS), 'E1 interval replay')
    _keys([d['row'] for d in records], ('family', 'iid', 'seed', 'budget'), 'E1 replay identity')
    rows, fixed = [], []
    for d in records:
        r, players = d['row'], d['players']
        _check(Path(d['_source']).stem == r['run_id'], f'Replay filename and run identity disagree: {d["_source"]}')
        _keys(players, ('player',), 'E1 player', ((i,) for i in range(r['d'])))
        has_reference = r['family'] != 'paper54'
        _check(r['has_reference'] == has_reference, f'Incorrect reference classification: {r["run_id"]}')
        _check(all((p['phi_exact'] is not None and math.isfinite(p['phi_exact'])) == has_reference for p in players), f'Missing or unexpected player reference: {r["run_id"]}')
        ref = players if has_reference else []
        delta = r['m_stop_b2'] - r['m_stop_a'] if r['eps_a'] is not None else None
        _check(delta == r.get('delta_m_b2'), f'Stored stopping delta is inconsistent: {r["run_id"]}')
        rows.append({'family': r['family'], 'iid': r['iid'], 'seed': r['seed'], 'budget': r['budget'], 'ref_n': len(ref), 'cover': all(p['L_b2'] <= p['phi_exact'] <= p['R_b2'] for p in ref) if ref else None, 'stop_delta': delta, 'xhat_differ': r['xhat_n_differ_b2'], 'xhat_total': r['xhat_n_total_b2'], 'xhat_max_diff': r['xhat_max_abs_diff_b2'], 'endpoint_diff': max(r['max_abs_L_b1_minus_L_b1p'], r['max_abs_R_b1_minus_R_b1p'])})
        if r['family'] == 'e11_312':
            original = source.json(r['record'])
            _check(original['instance_id'] == r['iid'] and original['seed'] == r['seed'] and original['budget'] == r['budget'] and original['arm'] == 'ours_cs', f'E11 replay/source identity mismatch: {r["run_id"]}')
            fixed.append({'cover': all(p['L_a'] <= p['phi_exact'] <= p['R_a'] for p in players), 'budget': r['budget'], 'iid': r['iid'], 'seed': r['seed'], 'beta': r['beta_a']})
    frame = pd.DataFrame(rows)
    expected_counts = {family: count for family, count in {'paper126': len(reference_ids) * len(PAPER_SEEDS), 'paper54': len(_primary_ids() - reference_ids) * len(PAPER_SEEDS), 'e11_312': 13 * 8 * len(PAPER_SEEDS), 'overcooked3': len(PAPER_SEEDS)}.items() if count}
    _check(frame.groupby('family').size().to_dict() == expected_counts, 'Incomplete E1 family cohorts')
    for family, ids, budgets in [('paper126', reference_ids, (None,)), ('paper54', _primary_ids() - reference_ids, (None,)), ('overcooked3', {'overcooked_cramped_room_k10'}, (None,)), ('e11_312', {r['iid'] for r in fixed}, tuple(2**k for k in range(9, 17)))]:
        group = [d['row'] for d in records if d['row']['family'] == family]
        _keys(group, ('iid', 'seed', 'budget'), f'E1 {family}', itertools.product(ids, PAPER_SEEDS, budgets))
    _count({r['iid'] for r in fixed}, 13, 'E11 table')
    interval = {'runs': len(frame), 'ref_runs': int(frame.ref_n.gt(0).sum()), 'ref_intervals': int(frame.ref_n.sum()), 'covered_runs': int(frame.cover.eq(True).sum()), 'stop_compared': int(frame.stop_delta.notna().sum()), 'stop_changes': int(frame.stop_delta.dropna().ne(0).sum()), 'xhat_differ': int(frame.xhat_differ.sum()), 'xhat_total': int(frame.xhat_total.sum()), 'max_xhat_diff': float(frame.xhat_max_diff.max()), 'max_endpoint_diff': float(frame.endpoint_diff.max()), 'by_family': frame.groupby('family').agg(runs=('family', 'size'), references=('ref_n', lambda x: (x > 0).sum()), intervals=('ref_n', 'sum')).to_dict('index')}
    ef = pd.DataFrame(fixed)
    replayed = {'runs': len(ef), 'covered': int(ef.cover.sum()), 'tasks': int(ef.iid.nunique()), 'seeds': sorted(ef.seed.unique().tolist()), 'budgets': sorted(ef.budget.unique().tolist()), 'beta': sorted(ef.beta.unique().tolist())}
    return {'E1': interval, 'E11_original_replayed': replayed}


def _numerical_audit(source):
    valid = source.csv('rebuttal/results/E1/input_validity.csv')
    _keys(valid.to_dict('records'), ('family', 'model_id'), 'E1 input-validity model')
    _count(valid, 188, 'E1 input validity')
    _check(valid.groupby('family').size().to_dict() == {'primary': 60, 'loo90': 90, 'tables13': 13, 'prism9': 9, 'decision6': 6, 'storm6': 6, 'overcooked': 4}, 'Incomplete E1 input-validity families')
    validity = {'models': len(valid), 'distinct_games': int(valid.game_key.nunique()), 'finite': bool(valid.finite_P.all() and valid.finite_pi.all()), 'negative_probabilities': int(valid.n_negative_P.sum() + valid.n_negative_pi.sum()), 'absorbing': bool(valid.fail_structurally_absorbing.all()), 'max_normalization_residual': float(max(valid.max_row_sum_dev.max(), valid.max_policy_sum_dev.max()))}
    outward = source.csv('rebuttal/results/E1/outward_bound.csv')
    _keys(outward.to_dict('records'), ('family', 'model_id'), 'E1 outward-bound model')
    _count(outward, 89, 'E1 outward-bound')
    outward_stats = {'model_rows': len(outward), 'exact_bound_larger_than_code': int(outward.eps_x_exact_minus_code.gt(0).sum()), 'max_exact_minus_code': float(outward.eps_x_exact_minus_code.max())}
    audit = source.csv('rebuttal/results/E1/precision_audit.csv')
    expected = _keys(audit.to_dict('records'), ('model_id', 'mask_kind'), 'E1 precision model/mask')
    _count(audit, 40, 'E1 precision audit')
    _check(np.finfo(np.longdouble).nmant > np.finfo(np.float64).nmant, 'Precision audit requires a platform with extended-precision numpy.longdouble')
    precision = []
    for path in sorted((source.root / 'rebuttal/results/E1/raw/precision').glob('*__ext.npz')):
        source.require(path.relative_to(source.root))
        gpu_path = source.require(path.with_name(path.name.replace('__ext.npz', '__gpu.npz')).relative_to(source.root))
        with np.load(path, allow_pickle=False) as ext, np.load(gpu_path, allow_pickle=False) as gpu:
            identity = (str(ext['iid'].item()), str(ext['kind'].item()))
            _check(identity in expected, f'Unexpected precision model/mask: {identity}')
            reference = ext['v_ld_hi'].astype(np.longdouble) + ext['v_ld_lo'].astype(np.longdouble)
            observed = gpu['v_gpu'].astype(np.longdouble)
            _check(reference.shape == observed.shape and np.isfinite(reference).all() and np.isfinite(observed).all(), f'Invalid precision arrays: {path.name}')
            _check(np.array_equal(ext['masks'], gpu['masks']), f'Precision coalition masks differ: {path.name}')
            _check(len(reference) == expected[identity]['n_coalitions'] and len(ext['mp_idx']) == expected[identity]['n_mp'], f'Incomplete precision cohort: {path.name}')
            precision.append({'model_id': identity[0], 'mask_kind': identity[1], 'n': len(reference), 'max_error': float(np.max(abs(observed - reference))), 'keys': ext.files, 'mp_n': len(ext['mp_idx'])})
    _keys(precision, ('model_id', 'mask_kind'), 'E1 precision binary pair', expected)
    binary = {'rows': len(precision), 'coalitions': sum(r['n'] for r in precision), 'max_error': max(r['max_error'] for r in precision), 'example_keys': precision[0]['keys'], 'mp_values': sum(r['mp_n'] for r in precision)}
    active = audit[audit.family.isin(['primary', 'decision6', 'tables13', 'overcooked'])]
    counts = {'model_ids': int(audit.model_id.nunique()), 'primary_models': len(set(audit.model_id) & _primary_ids()), 'mp_values': int(audit.n_mp.sum()), 'min_bound_to_empirical_scale': float((active.fp_term_eps_X / active.implied_bound_Z_minus_D).min())}
    return {'E1_validity': validity, 'E1_outward': outward_stats, 'E1_precision_binary': binary, 'E1_precision_counts': counts}


def _native(value):
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_native(item) for item in value]
    if isinstance(value, np.generic):
        return _native(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def generate(root: Path, out: Path) -> dict:
    source = _Inputs(root)
    stats, reference_ids = _timing(source)
    accuracy_ids = {iid for iid in reference_ids if int(iid.split('_')[2][1:]) <= 28}
    stats.update(_accuracy(source, accuracy_ids))
    stats['E7'] = _stopping(source, accuracy_ids)
    stats.update(_decision(source))
    stats['LOO'] = _loo(source)
    stats.update(_interval_audit(source, {iid for iid, arm in _exact_references(source)}))
    stats.update(_numerical_audit(source))
    stats['scope'] = {'original_sampler_seeds': list(PAPER_SEEDS), 'additional_sampler_seeds': list(NEW_SEEDS), 'top1_tie_absolute_tolerance': 1e-12, 'guarantee_conditional': True, 'exhaustive_references_fp64': True, 'interval_endpoints_from_stored_replay': True, 'prediction_difference_counters_from_stored_replay': True, 'precision_values_from_stored_npz': True, 'input_validity_from_recorded_audit': True, 'outward_bounds_from_recorded_audit': True, 'fresh_gpu_evaluations': 0}
    stats['input_files'] = sorted(source.paths)
    stats = _native(stats)
    table1 = []
    for iid in _decision_ids():
        r = stats['original_dt']['table1'][iid]
        table1.append({'instance_id': iid, 'agents': r['n'], 'states': r['states'], 'steps': r['k'], 'players': r['players'], 'absolute_radius': .005, 'sampler_runs': len(PAPER_SEEDS), 'certified_runs': r['certified_runs'], 'median_coalition_queries': int(r['queries']), 'median_gpu_time_s': r['time'], 'reachable_states': r['reachable'], 'positive_lower_bounds_min': r['positive_lower_min'], 'positive_lower_bounds_max': r['positive_lower_max']})
    table2 = []
    for div in DIVISORS:
        r = stats['E2B_table2'][str(div)]
        accuracy = stats['E2A_new'].get(f'ticketbench_{div}')
        table2.append({'tolerance_divisor': div, 'timing_runs': r['runs'], 'timing_tasks': r['tasks'], 'certified_runs': r['certified_runs'], 'median_gpu_time_s': r['median_time'], 'error_reference_runs': r['errors'], 'median_max_error_pct_of_upsilon': r['median_error_pct'], 'largest_max_error_pct_of_upsilon': r['max_error_pct'], 'top1_correct_new_seeds': accuracy['top1'] if accuracy else None, 'top1_runs_new_seeds': accuracy['runs'] if accuracy else None, 'top1_accuracy_pct_new_seeds': 100 * accuracy['top1'] / accuracy['runs'] if accuracy else None, 'cpu_exhaustive_matched_runs': r['cpu_ratios'], 'median_cpu_exhaustive_speedup': r['cpu_median_ratio'], 'gpu_exhaustive_matched_runs': r['gpu_ratios'], 'median_gpu_exhaustive_speedup': r['gpu_median_ratio'], 'sampled_runs_slower_than_exhaustive_gpu': r['gpu_slower']})
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(table1).to_csv(out / 'table1.csv', index=False)
    pd.DataFrame(table2).astype({'top1_correct_new_seeds': 'Int64', 'top1_runs_new_seeds': 'Int64'}).to_csv(out / 'table2.csv', index=False)
    return stats

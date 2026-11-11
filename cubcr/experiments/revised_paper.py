from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SEEDS = {20260829, 20260830, 20260831}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def median_or_none(values):
    values = [v for v in values if v is not None]
    return median(values) if values else None


def value_range(values):
    values = [v for v in values if v is not None]
    return [min(values), max(values)] if values else [None, None]


def measured_time(record):
    require(record.get('status') in ('OK', 'TO', 'NOT RUN', 'CRASH', 'MO'), 'Unknown run status')
    if record['status'] != 'OK' or record.get('censored'):
        return None
    wall = record.get('wall_s')
    require(wall is not None and math.isfinite(wall) and wall > 0, 'Completed run needs a positive finite wall time')
    return wall


def summarize_primary(root, out):
    from experiments.analyze import pair_cells, check_pair
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    input_files = set()

    def read(path):
        input_files.add(path.relative_to(root).as_posix())
        return json.loads(path.read_text())

    gpu = {}
    planned = set()
    for path in sorted((root / 'results/e9v3').glob('*__cert_gpu_s*.json')):
        d = read(path)
        if d.get('concurrent_load') or d.get('gen_seed') not in SEEDS or d.get('seed') not in SEEDS:
            continue
        key = (d['instance_id'], d['seed'])
        require(key not in gpu, f'Duplicate primary run: {key}')
        gpu[key] = d
        planned.add((d['n'], d['cell']['states'], d['k'], d['gen_seed'], d['seed']))
    expected = {(n, states, k, gs, seed) for n in (16, 20, 24, 28, 32) for states in (1000, 10000)
                for k in (50, 100) for gs in SEEDS for seed in SEEDS}
    require(planned == expected and len(gpu) == len(expected), 'Incomplete primary configuration grid')
    groups, combined, completed = defaultdict(list), [], set()
    status, primary_statuses = {}, defaultdict(int)
    for (iid, seed), d in sorted(gpu.items()):
        cpu = read(root / f'results/e9v3/e9v3__{iid}__cert_cpu_s{seed}.json')
        require(all(cpu[field] == d[field] for field in ('instance_id', 'n', 'k', 'gen_seed', 'seed')),
                f'Unmatched primary inputs: {iid}, {seed}')
        c, g = measured_time(cpu), measured_time(d)
        primary_statuses[f"cpu={cpu['status']},gpu={d['status']}"] += 1
        if c is not None and g is not None:
            for field in ('batch_sizes', 'n_solves', 'n_perms', 'n_units', 'shapley_sampled'):
                require(cpu[field] == d[field], f'Unmatched {field}: {iid}, {seed}')
        exact_path = root / f'results/e9v3/e9v3__{iid}__exact_cpu.json'
        exact = read(exact_path)
        timeout_path = exact_path.with_name(exact_path.stem + '_genuineTO.json')
        if timeout_path.exists():
            timeout = read(timeout_path)
            measured_time(timeout)
            if exact['status'] != 'OK':
                exact = timeout
        status[iid] = exact['status']
        exact_time = measured_time(exact)
        ratio = exact_time / g if exact_time is not None and g is not None else None
        if ratio is not None:
            combined.append(ratio)
            completed.add(iid)
        groups[(d['n'], d['cell']['states'], d['k'])].append((iid, seed, c, g, ratio))
    require(len(groups) == 20 and all(len(v) == 9 for v in groups.values()), 'Incomplete primary configuration grid')
    rows = []
    for (n, states, k), values in sorted(groups.items()):
        task_times = defaultdict(list)
        for iid, seed, c, g, _ in values:
            task_times[iid].append((seed, c, g))
        require(len(task_times) == 3 and all({v[0] for v in xs} == SEEDS for xs in task_times.values()), 'Incomplete matched task seeds')
        matched = [v for v in values if v[2] is not None and v[3] is not None]
        task_ratios = []
        for xs in task_times.values():
            xs = [v for v in xs if v[1] is not None and v[2] is not None]
            if xs:
                task_ratios.append(median(v[1] for v in xs) / median(v[2] for v in xs))
        ratios = [v[4] for v in values if v[4] is not None]
        rows.append({'n': n, 'states_target': states, 'k': k,
                     'exhaustive_cpu_over_sampled_gpu': median(ratios) if ratios else None,
                     'sampled_cpu_over_gpu_nested_median': median_or_none(task_ratios),
                     'sampled_cpu_over_gpu_matched_median': median_or_none(v[2] / v[3] for v in matched),
                     'sampled_gpu_median_s': median_or_none(v[3] for v in values),
                     'planned_runs': len(values), 'matched_runs': len(matched), 'exhaustive_ratios': len(ratios)})
    write_csv(out / 'figure3_data.csv', rows)
    e1 = [read(p) for p in sorted((root / 'results/e1').glob('*.json'))]
    seen = set()
    for d in e1:
        key = (d.get('cell_id') or d['instance_id'], d['engine'], d.get('threads'))
        require(key not in seen, f'Duplicate identical-batch record: {key}')
        seen.add(key)
    batch_groups = defaultdict(list)
    batch_seeds = defaultdict(set)
    for cell, arms in pair_cells(e1).items():
        g, c = arms.get('gpu'), arms.get('cpu32')
        require(g is not None and c is not None, f'Missing identical-batch arm: {cell}')
        scale = 10 ** round(math.log10(g['states']))
        key = (g['n'], scale, g['k'])
        require(g['seed'] == c['seed'] and g['seed'] not in batch_seeds[key], f'Duplicate or unmatched identical-batch seed: {cell}')
        batch_seeds[key].add(g['seed'])
        gt, ct = measured_time(g), measured_time(c)
        ratio = None
        if gt is not None and ct is not None:
            require(not check_pair(g, c), f'Invalid identical-batch pair: {cell}')
            ratio = ct / gt
        batch_groups[key].append(ratio)
    expected_cells = {(n, s, k) for n in (8, 12, 16, 20, 24) for s in (1000, 10000, 100000, 1000000) for k in (25, 100, 400)}
    require(set(batch_groups) == expected_cells and all(len(v) == 3 for v in batch_groups.values()) and all(v == SEEDS for v in batch_seeds.values()), 'Incomplete identical-batch grid')
    batch_rows = [{'n': n, 'states_target': states, 'k': k, 'cpu_over_gpu': median_or_none(v),
                   'runs': sum(x is not None for x in v), 'planned_runs': len(v)}
                  for (n, states, k), v in sorted(batch_groups.items())]
    write_csv(out / 'figure4_engine_data.csv', batch_rows)
    e2_groups = defaultdict(dict)
    for p in sorted((root / 'results/e2').glob('*.json')):
        d = read(p)
        scale = 10 ** round(math.log10(d['states']))
        cell = (d['n'], scale, d['k'])
        require(d['engine'] not in e2_groups[cell], f'Duplicate batching record: {cell}')
        e2_groups[cell][d['engine']] = d
    expected_batching = {(n, states, k) for n in (8, 12) for states in (1000, 10000) for k in (25, 100)}
    require(set(e2_groups) == expected_batching, 'Incomplete batching settings')
    batching_rows = []
    for (n, states, k), v in sorted(e2_groups.items()):
        require(set(v) == {'gpu_unbatched', 'gpu_batched'}, 'Incomplete batching arms')
        a, b = v['gpu_unbatched'], v['gpu_batched']
        at, bt = measured_time(a), measured_time(b)
        if at is not None and bt is not None:
            require(all(a[field] == b[field] for field in ('instance_id', 'mask_sha256', 'n_coalitions', 'dtype')), f'Unmatched batching work: {n}, {states}, {k}')
            require(a['n_coalitions'] == 2 ** n and a['full_enumeration'] and b['full_enumeration'], 'Batching experiment must enumerate all coalitions')
        batching_rows.append({'n': n, 'states_target': states, 'k': k,
                              'unbatched_s': at, 'batched_s': bt, 'unbatched_status': a['status'], 'batched_status': b['status'],
                              'speedup': at / bt if at is not None and bt is not None else None})
    write_csv(out / 'figure4_batching_data.csv', batching_rows)
    heatmap = defaultdict(list)
    for (n, states, k), vals in batch_groups.items():
        heatmap[(n, states)].extend(vals)
    return {'combined_speedup': median_or_none(combined), 'combined_ratios': len(combined),
            'tasks_with_cpu_reference': len(completed), 'primary_runs_per_engine': len(gpu),
            'primary_matched_completed_runs': sum(r['matched_runs'] for r in rows),
            'primary_pair_status_counts': dict(primary_statuses),
            'exhaustive_cpu_status_counts': {s: list(status.values()).count(s) for s in sorted(set(status.values()))},
            'cpu_gpu_range': value_range(r['sampled_cpu_over_gpu_nested_median'] for r in rows),
            'cpu_gpu_aggregation': 'median_across_tasks_of_ratio_of_median_matched_completed_sampler_times',
            'identical_batch_speedup': median_or_none(median_or_none(v) for v in batch_groups.values()),
            'batching_speedup_range': value_range(r['speedup'] for r in batching_rows),
            'heatmap': [{'n': n, 'states_target': states, 'cpu_over_gpu': median_or_none(v)} for (n, states), v in sorted(heatmap.items())],
            'input_files': sorted(input_files)}


def figures(out, primary):
    import math
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    with (out / 'figure3_data.csv').open() as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.7), layout='constrained')
    for states, color in [(1000, '#0072B2'), (10000, '#009E73')]:
        for k, marker in [(50, 'o'), (100, 's')]:
            selected = [r for r in rows if int(r['states_target']) == states and int(r['k']) == k]
            label = f'$|S|\\approx10^{int(math.log10(states))}$, $k={k}$'
            for ax, field in zip(axes, ['exhaustive_cpu_over_sampled_gpu', 'sampled_cpu_over_gpu_nested_median']):
                measured = [r for r in selected if r[field]]
                ax.plot([int(r['n']) for r in measured], [float(r[field]) for r in measured], color=color, marker=marker, linestyle='-' if k == 50 else '--', label=label, markersize=4)
    axes[0].set_yscale('log')
    axes[0].set_title('(a) Exhaustive CPU / sampled GPU')
    axes[1].set_title('(b) Sampled CPU / sampled GPU')
    for ax in axes:
        ax.set_xlabel('Agents')
        ax.set_ylabel('Speedup')
        ax.set_xticks([16, 20, 24, 28, 32])
        ax.grid(alpha=.25)
    axes[1].legend(fontsize=6)
    for suffix in ('pdf', 'png'):
        fig.savefig(out / f'figure3.{suffix}', dpi=250, metadata={'CreationDate': None, 'ModDate': None} if suffix == 'pdf' else {})
    plt.close(fig)
    heatmap = primary['heatmap']
    ns = sorted({r['n'] for r in heatmap})
    states = sorted({r['states_target'] for r in heatmap})
    values = np.array([[next(r['cpu_over_gpu'] for r in heatmap if r['n'] == n and r['states_target'] == s) for s in states] for n in ns], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.7), layout='constrained')
    im = axes[0].imshow(values, cmap='Blues', vmin=0, aspect='auto')
    for (i, j), v in np.ndenumerate(values):
        axes[0].text(j, i, f'{v:.1f}×' if np.isfinite(v) else '—', ha='center', va='center', color='white' if v > 4 else 'black')
    axes[0].set_xticks(range(len(states)), [f'$10^{int(math.log10(s))}$' for s in states])
    axes[0].set_yticks(range(len(ns)), ns)
    axes[0].set_xlabel('States')
    axes[0].set_ylabel('Agents')
    axes[0].set_title('(a) Identical batches: CPU / GPU')
    fig.colorbar(im, ax=axes[0], fraction=.04, label='Speedup')
    with (out / 'figure4_batching_data.csv').open() as f:
        batching = list(csv.DictReader(f))
    x = np.arange(len(batching))
    axes[1].bar(x-.2, [float(r['unbatched_s']) if r['unbatched_s'] else np.nan for r in batching], .4, label='One coalition per launch', color='#E69F00')
    axes[1].bar(x+.2, [float(r['batched_s']) if r['batched_s'] else np.nan for r in batching], .4, label='Batched', color='#0072B2')
    for i, r in enumerate(batching):
        if r['speedup']:
            axes[1].text(i, float(r['unbatched_s'])*1.25, f"{float(r['speedup']):.0f}×", ha='center', fontsize=6)
    axes[1].set_xticks(x, [f"{r['n']}\n$10^{int(math.log10(int(r['states_target'])))}$\n{r['k']}" for r in batching], fontsize=6)
    axes[1].set_xlabel('Agents / states / horizon')
    axes[1].set_ylabel('Seconds')
    if not any(r['unbatched_s'] or r['batched_s'] for r in batching):
        axes[1].set_ylim(1, 10)
        axes[1].text(.5, .5, 'No completed timings', transform=axes[1].transAxes, ha='center')
    axes[1].set_yscale('log')
    axes[1].set_title('(b) Same-GPU batching')
    axes[1].legend(fontsize=6)
    for suffix in ('pdf', 'png'):
        fig.savefig(out / f'figure4.{suffix}', dpi=250, metadata={'CreationDate': None, 'ModDate': None} if suffix == 'pdf' else {})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Regenerate revised-paper tables and measured figures from freshly generated local experiment records.')
    parser.add_argument('--output', type=Path, default=ROOT / 'paper/revised')
    parser.add_argument('--no-figures', action='store_true')
    args = parser.parse_args()
    from experiments import revised_statistics, revised_systems
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    primary = summarize_primary(ROOT, out)
    statistics = revised_statistics.generate(ROOT, out)
    systems = revised_systems.generate(ROOT, out)
    metrics = {'primary': primary, 'statistics': statistics, 'systems': systems}
    inputs = set()
    for section in metrics.values():
        inputs.update(section.pop('input_files', []))
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False) + '\n')
    (out / 'inputs.json').write_text(json.dumps([{'path': p, 'sha256': hashlib.sha256((ROOT / p).read_bytes()).hexdigest()} for p in sorted(inputs)], indent=2) + '\n')
    if not args.no_figures:
        figures(out, primary)
    print(f'Tables 1–4 and numerical summaries written to {out}')
    if not args.no_figures:
        print('Figures 3–4 written as PDF and PNG.')


if __name__ == '__main__':
    main()

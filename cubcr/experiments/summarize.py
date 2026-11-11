from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from experiments.protocol import REQUIRED_FIELDS


GROUP_KEYS = ["engine", "n", "states", "k", "dtype", "threads"]
TABLE_COLUMNS = ["engine", "n", "|S|", "k", "dtype", "threads", "throughput (coal/s)",
                 "speedup vs CPU", "max |err| vs FP64", "Kendall tau", "runs", "censored", "failed"]


def load_results(results_dir) -> pd.DataFrame:


    results_dir = Path(results_dir)
    rows = []
    if results_dir.is_dir():
        for p in sorted(results_dir.glob("*.json")):
            try:
                with open(p) as fh:
                    rec = json.load(fh)
            except (OSError, json.JSONDecodeError) as exc:
                print(f"[summarize] skipping {p.name}: {exc}", file=sys.stderr)
                continue
            if not isinstance(rec, dict) or any(f not in rec for f in REQUIRED_FIELDS):
                missing = [f for f in REQUIRED_FIELDS if not isinstance(rec, dict) or f not in rec]
                print(f"[summarize] skipping {p.name}: not a result record (missing {missing})",
                      file=sys.stderr)
                continue
            flat = {k: v for k, v in rec.items() if not isinstance(v, (dict, list))}
            flat["file"] = p.name
            rows.append(flat)
    cols = list(REQUIRED_FIELDS) + ["status", "speedup_is_lower_bound", "file"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df["status"] = df["status"].fillna("OK")
    df["censored"] = df["censored"].astype(bool)
    df["speedup_is_lower_bound"] = df["speedup_is_lower_bound"].fillna(False).astype(bool)
    return df


def _median(series) -> float | None:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    return float(np.median(vals)) if len(vals) else None


def aggregate(df: pd.DataFrame) -> pd.DataFrame:

    out_cols = GROUP_KEYS + ["throughput", "speedup", "speedup_lower_bound", "max_err",
                             "kendall_tau", "runs", "censored", "failed", "cap_s"]
    if df.empty:
        return pd.DataFrame(columns=out_cols)
    recs = []
    for key, g in df.groupby(GROUP_KEYS, dropna=False, sort=True):
        cens = g[g["censored"]]
        failed = g[(~g["censored"]) & (g["status"] != "OK")]
        ok = g[(~g["censored"]) & (g["status"] == "OK")]
        rec = dict(zip(GROUP_KEYS, key))
        rec["throughput"] = _median(ok["coalition_solves_per_s"]) if len(ok) else None
        rec["speedup"] = _median(ok["speedup_vs_cpu"]) if len(ok) else None
        rec["speedup_lower_bound"] = bool(ok["speedup_is_lower_bound"].any()) if len(ok) else False
        rec["max_err"] = (float(pd.to_numeric(ok["max_abs_err_vs_fp64"], errors="coerce").max())
                          if len(ok) and pd.to_numeric(ok["max_abs_err_vs_fp64"], errors="coerce").notna().any()
                          else None)
        rec["kendall_tau"] = _median(ok["kendall_tau"]) if len(ok) else None
        rec["runs"] = int(len(g))
        rec["censored"] = int(len(cens))
        rec["failed"] = int(len(failed))
        rec["cap_s"] = _median(cens["cap_s"]) if len(cens) else _median(g["cap_s"])
        recs.append(rec)
    return pd.DataFrame(recs, columns=out_cols)


def _fmt_num(x, fmt) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return format(x, fmt)


def _fmt_int(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return str(int(x))


def table_rows(agg: pd.DataFrame) -> list[list[str]]:
    rows = []
    for _, r in agg.iterrows():
        all_censored = r["runs"] > 0 and r["censored"] == r["runs"]
        if all_censored:
            thr = f"TO (>= {r['cap_s']:g} s)" if r["cap_s"] is not None and not pd.isna(r["cap_s"]) else "TO"
            sp = "-"
        else:
            thr = _fmt_num(r["throughput"], ".4g")
            sp = _fmt_num(r["speedup"], ".2f")
            if sp != "-":
                sp = (">= " if r["speedup_lower_bound"] else "") + sp + "x"
        rows.append([
            str(r["engine"]), _fmt_int(r["n"]), _fmt_int(r["states"]), _fmt_int(r["k"]), str(r["dtype"]),
            _fmt_int(r["threads"]), thr, sp,
            _fmt_num(r["max_err"], ".2e"), _fmt_num(r["kendall_tau"], ".3f"),
            _fmt_int(r["runs"]), f"{int(r['censored'])}/{int(r['runs'])}", _fmt_int(r["failed"]),
        ])
    return rows


def to_markdown(agg: pd.DataFrame, title: str | None = None) -> str:

    lines = []
    if title:
        lines.append(f"## {title}")
        lines.append("")
    if agg.empty:
        lines.append("_No result records found (results directory empty or contains no valid *.json)._")
        return "\n".join(lines) + "\n"
    rows = table_rows(agg)

    header = [c.replace("|", "\\|") for c in TABLE_COLUMNS]
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]
    lines.append("| " + " | ".join(c.ljust(w) for c, w in zip(header, widths)) + " |")
    lines.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for r in rows:
        lines.append("| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |")
    lines.append("")
    lines.append("Medians over uncensored runs of each cell; `censored` = runs that hit their cap "
                 "(reported as `>= cap`, excluded from medians); `failed` = CRASH/MO/NOT RUN runs.")
    return "\n".join(lines) + "\n"


def summarize(results_dir, out_path=None, title="cuBCR results summary") -> str:
    df = load_results(results_dir)
    agg = aggregate(df)
    md = to_markdown(agg, title=title)
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md)
    return md


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(REPO_ROOT / "results"))
    ap.add_argument("--out", default=None, help="markdown output (default <results>/summary.md)")
    ap.add_argument("--no-write", action="store_true", help="print only")
    a = ap.parse_args(argv)
    out = None if a.no_write else (a.out or str(Path(a.results) / "summary.md"))
    md = summarize(a.results, out)
    print(md)
    if out:
        print(f"[summarize] wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()

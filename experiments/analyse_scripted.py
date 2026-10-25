"""Aggregate results/scripted_results.jsonl into tables and a figure."""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(__file__)
INP = os.path.join(HERE, "..", "results", "scripted_results.jsonl")
OUT_JSON = os.path.join(HERE, "..", "results", "scripted_summary.json")
OUT_FIG = os.path.join(HERE, "..", "results", "fig_scripted.png")

DETS = ["NI/none", "NI/prev_majority", "NI+perm/none", "NI+perm/prev_majority", "MI", "agree", "cramer"]
NI_DETS = [d for d in DETS if d.startswith("NI")]
LABELS = {"NI/none": "NI (memoryless)", "NI/prev_majority": "NI (state=prev majority)",
          "NI+perm/none": "NI+perm (memoryless)", "NI+perm/prev_majority": "NI+perm (state=prev majority)",
          "MI": "raw MI", "agree": "same-action rate", "cramer": "Cramer's V"}


def main():
    recs = [json.loads(l) for l in open(INP) if l.strip()]
    groups = defaultdict(list)
    for r in recs:
        groups[(r["experiment"], r["condition"], r["cp"], r["T"])].append(r)

    def mean(key, rs):
        vals = [r[key] for r in rs if key in r]
        return float(np.mean(vals)) if vals else None

    summary = {}
    for k, rs in sorted(groups.items()):
        exp, cond, cp, T = k
        row = {"experiment": exp, "condition": cond, "cp": cp, "T": T, "n": len(rs)}
        for d in DETS:
            row[f"{d}|top_true"] = mean(f"{d}|top_true", rs)
            row[f"{d}|top_innocent"] = mean(f"{d}|top_innocent", rs)
        for d in NI_DETS:
            row[f"{d}|flag_any"] = mean(f"{d}|flag_any", rs)
            row[f"{d}|flag_true"] = mean(f"{d}|flag_true", rs)
            row[f"{d}|flag_innocent"] = mean(f"{d}|flag_innocent", rs)
            row[f"{d}|score_true_mean"] = mean(f"{d}|score_true", rs)
            row[f"{d}|score_innocent_mean"] = mean(f"{d}|score_innocent", rs)
        summary[f"{exp}|{cond}|cp={cp}|T={T}"] = row
    json.dump(summary, open(OUT_JSON, "w"), indent=2)

    # ---- printed tables ----
    def table(title, rows, cols, fmt="{:.2f}"):
        print(f"\n{title}")
        print("T/CP".ljust(8) + "".join(c.rjust(26) for c in cols))
        for label, r in rows:
            print(str(label).ljust(8) + "".join((fmt.format(r[c]) if r.get(c) is not None else "-").rjust(26) for c in cols))

    Ts = [25, 50, 100, 200, 500]
    for cond in ("bonjour", "parallel"):
        rows = [(T, summary[f"A|{cond}|cp=0.6|T={T}"]) for T in Ts]
        table(f"Rank-1 accuracy of true pair, condition={cond}, CP=0.6", rows, [f"{d}|top_true" for d in DETS])
        table(f"Bonjour criterion: exactly-one-pair flagged AND correct, condition={cond}, CP=0.6", rows, [f"{d}|flag_true" for d in NI_DETS])
    for cond in ("null", "parallel_only"):
        rows = [(T, summary[f"A|{cond}|cp=0.0|T={T}"]) for T in Ts]
        table(f"False-positive rate (any pair flagged), condition={cond}", rows, [f"{d}|flag_any" for d in NI_DETS])
    rows = [(T, summary[f"A|parallel_only|cp=0.0|T={T}"]) for T in Ts]
    table("Innocent parallel pair flagged, condition=parallel_only", rows, [f"{d}|flag_innocent" for d in NI_DETS])
    for T in (25, 100):
        for cond in ("bonjour", "parallel"):
            rows = [(cp, summary[f"B|{cond}|cp={cp}|T={T}"]) for cp in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
            table(f"CP sweep, T={T}, condition={cond}: rank-1 accuracy", rows, [f"{d}|top_true" for d in DETS])
            table(f"CP sweep, T={T}, condition={cond}: criterion correct", rows, [f"{d}|flag_true" for d in NI_DETS])

    # ---- figure ----
    import matplotlib
    import matplotlib.ticker
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    colors = {"NI/none": "#b2182b", "NI/prev_majority": "#2166ac", "NI+perm/none": "#ef8a62",
              "NI+perm/prev_majority": "#67a9cf", "MI": "#999999", "agree": "#4dac26", "cramer": "#7b3294"}
    styles = {"MI": ":", "agree": "--", "cramer": "-."}
    ax = axes[0]
    for d in DETS:
        ax.plot(Ts, [summary[f"A|parallel|cp=0.6|T={T}"][f"{d}|top_true"] for T in Ts], marker="o",
                color=colors[d], ls=styles.get(d, "-"), label=LABELS[d])
    ax.set_xscale("log"); ax.set_xticks(Ts); ax.set_xticklabels(Ts); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(-0.03, 1.03); ax.set_xlabel("rounds T"); ax.set_ylabel("rank-1 accuracy of true pair")
    ax.set_title("Colluding pair + innocent parallel pair (CP=0.6)")
    ax.legend(fontsize=7, loc="lower right")
    ax = axes[1]
    for d in NI_DETS:
        ax.plot(Ts, [summary[f"A|parallel_only|cp=0.0|T={T}"][f"{d}|flag_any"] for T in Ts], marker="s",
                color=colors[d], label=LABELS[d] + " (parallel-only)")
        ax.plot(Ts, [summary[f"A|null|cp=0.0|T={T}"][f"{d}|flag_any"] for T in Ts], marker="^", ls="--",
                color=colors[d], label=LABELS[d] + " (all random)")
    ax.set_xscale("log"); ax.set_xticks(Ts); ax.set_xticklabels(Ts); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(-0.03, 1.03); ax.set_xlabel("rounds T"); ax.set_ylabel("false-positive rate (criterion, alpha=0.05)")
    ax.set_title("No collusion present")
    ax.legend(fontsize=6, loc="center right")
    ax = axes[2]
    cps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for d in NI_DETS:
        ax.plot(cps, [summary[f"B|bonjour|cp={cp}|T=25"][f"{d}|flag_true"] for cp in cps], marker="o",
                color=colors[d], label=LABELS[d])
    ax.set_ylim(-0.03, 1.03); ax.set_xlabel("collusion probability CP"); ax.set_ylabel("criterion correct")
    ax.set_title("T=25 rounds (LLM-pilot scale), random non-colluders")
    ax.legend(fontsize=7, loc="upper left")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=160)
    print("\nfigure saved", OUT_FIG)


if __name__ == "__main__":
    main()

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
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "font.size": 11, "axes.titlesize": 12,
                         "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
                         "legend.fontsize": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    # one colour per detector, shared across every panel
    colors = {"NI/none": "#d9822b", "NI+perm/none": "#f2b77a",
              "NI/prev_majority": "#2c6fac", "NI+perm/prev_majority": "#8ab4dc",
              "MI": "#777777", "agree": "#444444", "cramer": "#aaaaaa"}
    styles = {"MI": ":", "agree": "--", "cramer": "-."}
    markers = {"MI": "v", "agree": "D", "cramer": "x"}
    lw = 2.0
    ax = axes[0]
    for d in DETS:
        ax.plot(Ts, [summary[f"A|parallel|cp=0.6|T={T}"][f"{d}|top_true"] for T in Ts], marker=markers.get(d, "o"),
                ms=7, lw=lw, color=colors[d], ls=styles.get(d, "-"), label=LABELS[d])
    ax.set_xscale("log"); ax.set_xticks(Ts); ax.set_xticklabels(Ts); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(-0.03, 1.03); ax.set_xlabel("rounds T"); ax.set_ylabel("rank-1 accuracy of the true pair")
    ax.set_title("(a) Innocent parallel pair present, CP = 0.6")
    ax.legend(loc="center right", frameon=False, handlelength=2.6)
    ax = axes[1]
    for d in NI_DETS:
        ax.plot(Ts, [summary[f"A|parallel_only|cp=0.0|T={T}"][f"{d}|flag_any"] for T in Ts], marker="s", ms=7, lw=lw,
                color=colors[d], label=LABELS[d] + " (parallel-only)")
        ax.plot(Ts, [summary[f"A|null|cp=0.0|T={T}"][f"{d}|flag_any"] for T in Ts], marker="^", ms=7, lw=lw, ls="--",
                color=colors[d], label=LABELS[d] + " (all random)")
    ax.set_xscale("log"); ax.set_xticks(Ts); ax.set_xticklabels(Ts); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(-0.03, 1.03); ax.set_xlabel("rounds T"); ax.set_ylabel("false-positive rate (alpha = 0.05)")
    ax.set_title("(b) No collusion present")
    # colour is given by the panel (a) legend; this legend only explains the two worlds
    world_handles = [Line2D([], [], color="#444444", marker="s", ms=7, lw=lw, ls="-", label="parallel-only world"),
                     Line2D([], [], color="#444444", marker="^", ms=7, lw=lw, ls="--", label="all-random world")]
    ax.legend(handles=world_handles, loc="center right", frameon=False, handlelength=2.6)
    ax = axes[2]
    cps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for d in NI_DETS:
        ax.plot(cps, [summary[f"B|bonjour|cp={cp}|T=25"][f"{d}|flag_true"] for cp in cps], marker="o", ms=7, lw=lw,
                color=colors[d], label=LABELS[d])
    ax.set_ylim(-0.03, 1.03); ax.set_xlabel("collusion probability CP"); ax.set_ylabel("criterion correct (true pair, alone)")
    ax.set_title("(c) T = 25, random non-colluders")
    ax.legend(loc="upper left", frameon=False, handlelength=2.6)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(OUT_FIG, dpi=160)
    print("\nfigure saved", OUT_FIG)


if __name__ == "__main__":
    main()

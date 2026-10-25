"""
Three extensions to the scripted RPS validation (experiments/scripted_rps.py).

Extension 1, calibrated criterion
  Replace the fixed alpha = 0.05 threshold with a p-value rule. For every unordered
  pair we take the within-state permutation p-value of gamma (from
  permutation_corrected_gamma), Holm-correct over the 6 pairs, and declare a pair
  colluding iff its Holm p-value < 0.05 AND its net-influence score
  (min of both directions, on corrected gamma) is the maximum among pairs.
  Conditions bonjour / parallel (power) and null / parallel_only (false positives),
  T in {25, 50, 100, 200}, CP = 0.6, state = prev_majority. The fixed-alpha rule is
  evaluated on the same games.

Extension 2, hard confound (parallel_rich)
  An innocent pair C, D that keys on a richer public state than the detector
  conditions on. Rule (deterministic, documented here):
    round 0: uniform random.
    round t > 0: score the previous round with the RPS payoff of the report
      (all same: nobody scores; all three actions present: everyone scores;
      exactly two actions present: holders of the winning action score).
      Let w be the LOWEST-INDEX player who scored in round t-1 (player 0 if
      nobody scored). Play the action that BEATS w's round t-1 action.
  This is a function of the previous full action profile but NOT of the previous
  majority alone (profiles RRPS and PRRS share majority R but give P and S).
  Conditions parallel_rich (A random, B assistant CP = 0.6, C and D parallel_rich)
  and parallel_rich_only (A, B random). T in {25, 50, 100, 200, 500}; state =
  none, prev_majority, prev_profile; with and without permutation correction.

Extension 3, coalition of three
  5 players. A random primary; B and C assistants that each play the action that
  loses to A with prob CP, else random; D, E random. True pairs AB, AC, BC.
  T in {50, 100, 200, 500}, CP = 0.6, state = none and prev_majority.
  Metrics: number of true pairs in the top 3 by score (net influence and raw
  conditional MI), how often the Bonjour exactly-one-pair criterion returns None,
  rank-1 accuracy of any true pair under NI vs raw MI, and the diagnostic
  gamma(B; C) unconditional vs conditional on A's current action.

Outputs: results/extensions_results.jsonl, results/extensions_summary.json,
results/extensions_tables.txt, results/fig_extensions.png.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from netinfluence.stats import (  # noqa: E402
    STATE_FUNCS, conditional_mi, net_influence, detect_pair, pair_scores,
    permutation_corrected_gamma, symmetric_top_pair,
)
from experiments.scripted_rps import simulate, loses_to, beats, N_ACTIONS, ALPHA  # noqa: E402

RESULTS = os.path.join(HERE, "..", "results")
OUT_JSONL = os.path.join(RESULTS, "extensions_results.jsonl")
OUT_SUMMARY = os.path.join(RESULTS, "extensions_summary.json")
OUT_TABLES = os.path.join(RESULTS, "extensions_tables.txt")
OUT_FIG = os.path.join(RESULTS, "fig_extensions.png")

TRUE_PAIR = (0, 1)
INNOCENT_PAIR = (2, 3)
COALITION_PAIRS = [(0, 1), (0, 2), (1, 2)]
CP = 0.6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def score_round(profile: np.ndarray) -> np.ndarray:
    """RPS payoff from report section 3.4: 1 point to each scoring player."""
    present = np.unique(profile)
    if len(present) == 1:
        return np.zeros(len(profile), dtype=np.int64)
    if len(present) == 3:
        return np.ones(len(profile), dtype=np.int64)
    x, y = present
    winner = x if beats(y) == x else y
    return (profile == winner).astype(np.int64)


def parallel_rich_action(prev_profile: np.ndarray) -> int:
    """Beat the previous action of the lowest-index player who scored last round
    (player 0 if nobody scored)."""
    sc = score_round(prev_profile)
    w = int(np.argmax(sc)) if sc.any() else 0
    return beats(int(prev_profile[w]))


def holm(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def calibrated_detect(G_corr: np.ndarray, pvals: np.ndarray, level: float = 0.05):
    """Extension 1 rule: Holm p < level AND pair is the arg-max net-influence score."""
    scores = pair_scores(G_corr)
    pairs = list(scores)
    p = np.array([pvals[i, j] for i, j in pairs])
    p_holm = holm(p)
    top = max(pairs, key=scores.get)
    k = pairs.index(top)
    return (top if p_holm[k] < level else None), {pr: float(ph) for pr, ph in zip(pairs, p_holm)}


# ---------------------------------------------------------------------------
# Simulators
# ---------------------------------------------------------------------------

def simulate_rich(condition: str, cp: float, T: int, rng: np.random.Generator) -> np.ndarray:
    roles = {
        "parallel_rich": ["random", "assistant", "parallel_rich", "parallel_rich"],
        "parallel_rich_only": ["random", "random", "parallel_rich", "parallel_rich"],
    }[condition]
    N = len(roles)
    acts = np.zeros((T, N), dtype=np.int64)
    for t in range(T):
        a0 = rng.integers(N_ACTIONS)
        for i, role in enumerate(roles):
            if role == "random":
                acts[t, i] = a0 if i == 0 else rng.integers(N_ACTIONS)
            elif role == "assistant":
                acts[t, i] = loses_to(a0) if rng.random() < cp else rng.integers(N_ACTIONS)
            elif role == "parallel_rich":
                acts[t, i] = rng.integers(N_ACTIONS) if t == 0 else parallel_rich_action(acts[t - 1])
    return acts


def simulate_coalition(cp: float, T: int, rng: np.random.Generator) -> np.ndarray:
    roles = ["random", "assistant", "assistant", "random", "random"]
    N = len(roles)
    acts = np.zeros((T, N), dtype=np.int64)
    for t in range(T):
        a0 = rng.integers(N_ACTIONS)
        for i, role in enumerate(roles):
            if role == "random":
                acts[t, i] = a0 if i == 0 else rng.integers(N_ACTIONS)
            else:
                acts[t, i] = loses_to(a0) if rng.random() < cp else rng.integers(N_ACTIONS)
    return acts


# ---------------------------------------------------------------------------
# Per-game jobs
# ---------------------------------------------------------------------------

def job_ext1(condition, cp, T, seed, n_perm):
    rng = np.random.default_rng(seed)
    acts = simulate(condition, cp, T, rng)
    s = STATE_FUNCS["prev_majority"](acts, N_ACTIONS)
    g, g_corr, pvals = permutation_corrected_gamma(acts, s, N_ACTIONS, n_perm=n_perm, rng=rng)
    rec = {"ext": 1, "condition": condition, "cp": cp, "T": T, "seed": seed, "state": "prev_majority"}
    # fixed alpha rules on the same game
    for label, gm in (("fixed/NI", g), ("fixed/NI+perm", g_corr)):
        flagged, scores = detect_pair(net_influence(gm), ALPHA)
        rec[f"{label}|flag_any"] = int(flagged is not None)
        rec[f"{label}|flag_true"] = int(flagged == TRUE_PAIR)
        rec[f"{label}|flag_innocent"] = int(flagged == INNOCENT_PAIR)
    # calibrated rule
    G_corr = net_influence(g_corr)
    declared, p_holm = calibrated_detect(G_corr, pvals, 0.05)
    rec["calib|flag_any"] = int(declared is not None)
    rec["calib|flag_true"] = int(declared == TRUE_PAIR)
    rec["calib|flag_innocent"] = int(declared == INNOCENT_PAIR)
    rec["calib|p_holm_true"] = p_holm[TRUE_PAIR]
    rec["calib|p_raw_true"] = float(pvals[TRUE_PAIR])
    rec["calib|min_p_holm"] = float(min(p_holm.values()))
    rec["calib|top_true"] = int(max(pair_scores(G_corr), key=pair_scores(G_corr).get) == TRUE_PAIR)
    return rec


def job_ext2(condition, cp, T, seed, n_perm):
    rng = np.random.default_rng(seed)
    acts = simulate_rich(condition, cp, T, rng)
    rec = {"ext": 2, "condition": condition, "cp": cp, "T": T, "seed": seed}
    rec["cd_agree"] = float(np.mean(acts[1:, 2] == acts[1:, 3]))
    for sname in ("none", "prev_majority", "prev_profile"):
        s = STATE_FUNCS[sname](acts, N_ACTIONS)
        rec[f"n_states/{sname}"] = int(len(np.unique(s)))
        g, g_corr, pvals = permutation_corrected_gamma(acts, s, N_ACTIONS, n_perm=n_perm, rng=rng)
        for label, gm in ((f"NI/{sname}", g), (f"NI+perm/{sname}", g_corr)):
            G = net_influence(gm)
            flagged, scores = detect_pair(G, ALPHA)
            top = max(scores, key=scores.get)
            rec[f"{label}|top_true"] = int(top == TRUE_PAIR)
            rec[f"{label}|top_innocent"] = int(top == INNOCENT_PAIR)
            rec[f"{label}|flag_any"] = int(flagged is not None)
            rec[f"{label}|flag_true"] = int(flagged == TRUE_PAIR)
            rec[f"{label}|flag_innocent"] = int(flagged == INNOCENT_PAIR)
            rec[f"{label}|score_true"] = scores[TRUE_PAIR]
            rec[f"{label}|score_innocent"] = scores[INNOCENT_PAIR]
            rec[f"{label}|gamma_innocent"] = float(gm[INNOCENT_PAIR])
        declared, _ = calibrated_detect(net_influence(g_corr), pvals, 0.05)
        rec[f"calib/{sname}|flag_any"] = int(declared is not None)
        rec[f"calib/{sname}|flag_true"] = int(declared == TRUE_PAIR)
        rec[f"calib/{sname}|flag_innocent"] = int(declared == INNOCENT_PAIR)
    return rec


def job_ext3(cp, T, seed, n_perm):
    rng = np.random.default_rng(seed)
    acts = simulate_coalition(cp, T, rng)
    rec = {"ext": 3, "condition": "coalition3", "cp": cp, "T": T, "seed": seed}
    pairs = list(itertools.combinations(range(acts.shape[1]), 2))
    for sname in ("none", "prev_majority"):
        s = STATE_FUNCS[sname](acts, N_ACTIONS)
        g, g_corr, pvals = permutation_corrected_gamma(acts, s, N_ACTIONS, n_perm=n_perm, rng=rng)
        for label, gm in ((f"NI/{sname}", g), (f"NI+perm/{sname}", g_corr)):
            G = net_influence(gm)
            flagged, scores = detect_pair(G, ALPHA)
            n_cross = sum(v >= ALPHA for v in scores.values())
            ranked = sorted(pairs, key=lambda p: -scores[p])
            rec[f"{label}|true_in_top3"] = int(sum(p in COALITION_PAIRS for p in ranked[:3]))
            rec[f"{label}|top_any_true"] = int(ranked[0] in COALITION_PAIRS)
            rec[f"{label}|top_is_BC"] = int(ranked[0] == (1, 2))
            rec[f"{label}|flag_none"] = int(flagged is None)
            rec[f"{label}|flag_none_multi"] = int(n_cross > 1)
            rec[f"{label}|flag_none_zero"] = int(n_cross == 0)
            rec[f"{label}|flag_true"] = int(flagged in COALITION_PAIRS)
            rec[f"{label}|n_cross"] = n_cross
            for p in COALITION_PAIRS:
                rec[f"{label}|score_{p[0]}{p[1]}"] = scores[p]
            # raw conditional MI (no net correction), same state
            mi_scores = {p: float(gm[p]) for p in pairs}
            ranked_mi = sorted(pairs, key=lambda p: -mi_scores[p])
            mlabel = label.replace("NI", "MI")
            rec[f"{mlabel}|true_in_top3"] = int(sum(p in COALITION_PAIRS for p in ranked_mi[:3]))
            rec[f"{mlabel}|top_any_true"] = int(ranked_mi[0] in COALITION_PAIRS)
            rec[f"{mlabel}|top_is_BC"] = int(ranked_mi[0] == (1, 2))
            rec[f"{mlabel}|top2_are_AB_AC"] = int(set(ranked_mi[:2]) == {(0, 1), (0, 2)})
            for p in COALITION_PAIRS:
                rec[f"{mlabel}|gamma_{p[0]}{p[1]}"] = mi_scores[p]
        if sname == "none":
            # diagnostic: gamma(B; C) conditional on A's current action
            rec["gamma_BC_given_A"] = conditional_mi(acts[:, 1], acts[:, 2], acts[:, 0], N_ACTIONS)
            rec["gamma_BC_raw"] = float(g[1, 2])
            rec["gamma_AB_raw"] = float(g[0, 1])
    return rec


def one_job(args):
    kind, rest = args[0], args[1:]
    return {1: job_ext1, 2: job_ext2, 3: job_ext3}[kind](*rest)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(recs):
    groups = defaultdict(list)
    for r in recs:
        groups[(r["ext"], r["condition"], r["T"])].append(r)
    summary = {}
    for (ext, cond, T), rs in sorted(groups.items()):
        keys = set().union(*(r.keys() for r in rs)) - {"ext", "condition", "cp", "T", "seed", "state"}
        row = {"ext": ext, "condition": cond, "T": T, "n": len(rs), "cp": rs[0]["cp"]}
        for k in sorted(keys):
            vals = [r[k] for r in rs if k in r]
            row[k] = float(np.mean(vals)) if vals else None
        summary[f"ext{ext}|{cond}|T={T}"] = row
    return summary


def write_tables(summary, fh):
    def table(title, rows, cols, fmt="{:.2f}"):
        width = max(len(c) for c in cols) + 2
        print(f"\n{title}", file=fh)
        print("T".ljust(6) + "".join(c.rjust(width) for c in cols), file=fh)
        for label, r in rows:
            print(str(label).ljust(6) + "".join((fmt.format(r[c]) if r.get(c) is not None else "-").rjust(width) for c in cols), file=fh)

    # ---- Ext 1 ----
    T1 = [25, 50, 100, 200]
    for cond in ("bonjour", "parallel"):
        rows = [(T, summary[f"ext1|{cond}|T={T}"]) for T in T1]
        table(f"Ext 1  power (declared pair == true pair), condition={cond}, CP=0.6, state=prev_majority", rows,
              ["fixed/NI|flag_true", "fixed/NI+perm|flag_true", "calib|flag_true", "calib|top_true", "calib|p_holm_true"])
    for cond in ("null", "parallel_only"):
        rows = [(T, summary[f"ext1|{cond}|T={T}"]) for T in T1]
        table(f"Ext 1  false-positive rate (any pair declared), condition={cond}, state=prev_majority", rows,
              ["fixed/NI|flag_any", "fixed/NI+perm|flag_any", "calib|flag_any", "calib|flag_innocent", "calib|min_p_holm"])

    # ---- Ext 2 ----
    T2 = [25, 50, 100, 200, 500]
    states = ["none", "prev_majority", "prev_profile"]
    rows = [(T, summary[f"ext2|parallel_rich|T={T}"]) for T in T2]
    table("Ext 2  rank-1 accuracy of true pair AB, condition=parallel_rich, CP=0.6", rows,
          [f"NI/{s}|top_true" for s in states] + [f"NI+perm/{s}|top_true" for s in states])
    table("Ext 2  innocent pair CD ranked first, condition=parallel_rich", rows,
          [f"NI/{s}|top_innocent" for s in states] + [f"NI+perm/{s}|top_innocent" for s in states])
    table("Ext 2  criterion correct (alpha=0.05), condition=parallel_rich", rows,
          [f"NI/{s}|flag_true" for s in states] + [f"NI+perm/{s}|flag_true" for s in states])
    table("Ext 2  calibrated (Holm p<0.05 + argmax) correct, condition=parallel_rich", rows,
          [f"calib/{s}|flag_true" for s in states])
    table("Ext 2  number of occupied states and mean gamma(C;D|S)", rows,
          [f"n_states/{s}" for s in states] + [f"NI/{s}|gamma_innocent" for s in states] + ["cd_agree"], fmt="{:.3f}")
    rows = [(T, summary[f"ext2|parallel_rich_only|T={T}"]) for T in T2]
    table("Ext 2  false-positive rate (any pair flagged, alpha=0.05), condition=parallel_rich_only", rows,
          [f"NI/{s}|flag_any" for s in states] + [f"NI+perm/{s}|flag_any" for s in states])
    table("Ext 2  innocent pair CD flagged (alpha=0.05), condition=parallel_rich_only", rows,
          [f"NI/{s}|flag_innocent" for s in states] + [f"NI+perm/{s}|flag_innocent" for s in states])
    table("Ext 2  calibrated rule: any pair declared / innocent declared, condition=parallel_rich_only", rows,
          [f"calib/{s}|flag_any" for s in states] + [f"calib/{s}|flag_innocent" for s in states])

    # ---- Ext 3 ----
    T3 = [50, 100, 200, 500]
    rows = [(T, summary[f"ext3|coalition3|T={T}"]) for T in T3]
    for s in ("none", "prev_majority"):
        table(f"Ext 3  coalition of three (A,B,C), CP=0.6, state={s}: true pairs in top 3 (of 3) and rank-1 any true pair", rows,
              [f"NI/{s}|true_in_top3", f"NI+perm/{s}|true_in_top3", f"MI/{s}|true_in_top3",
               f"NI/{s}|top_any_true", f"NI+perm/{s}|top_any_true", f"MI/{s}|top_any_true", f"MI/{s}|top2_are_AB_AC"])
        table(f"Ext 3  state={s}: Bonjour criterion returns None (total / because >1 pair crosses / because 0 cross) and correct", rows,
              [f"NI/{s}|flag_none", f"NI/{s}|flag_none_multi", f"NI/{s}|flag_none_zero", f"NI/{s}|flag_true",
               f"NI+perm/{s}|flag_none", f"NI+perm/{s}|flag_none_multi", f"NI+perm/{s}|flag_none_zero", f"NI+perm/{s}|flag_true"])
    table("Ext 3  state=none: mean raw gamma and net scores (nats); gamma(B;C) raw vs conditional on A", rows,
          ["MI/none|gamma_01", "MI/none|gamma_02", "MI/none|gamma_12", "gamma_BC_given_A",
           "NI/none|score_01", "NI/none|score_02", "NI/none|score_12", "MI/none|top_is_BC"], fmt="{:.3f}")


def make_figure(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    # validated categorical slots (dataviz reference palette, light mode)
    BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
    GRID = "#e4e3df"
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    def logx(ax, Ts):
        ax.set_xscale("log")
        ax.set_xticks(Ts)
        ax.set_xticklabels([str(t) for t in Ts])
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_xlabel("rounds T")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(color=GRID, lw=1)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    # Panel 1: calibrated vs fixed alpha
    ax = axes[0]
    T1 = [25, 50, 100, 200]
    def s1(cond, key): return [summary[f"ext1|{cond}|T={T}"][key] for T in T1]
    ax.plot(T1, s1("parallel", "calib|flag_true"), color=BLUE, lw=2, marker="o", ms=7, label="calibrated: power, parallel")
    ax.plot(T1, s1("parallel", "fixed/NI+perm|flag_true"), color=ORANGE, lw=2, marker="o", ms=7, label="fixed alpha: power, parallel")
    ax.plot(T1, s1("parallel_only", "calib|flag_any"), color=BLUE, lw=2, ls="--", marker="s", ms=7, label="calibrated: FPR, parallel-only")
    ax.plot(T1, s1("parallel_only", "fixed/NI+perm|flag_any"), color=ORANGE, lw=2, ls="--", marker="s", ms=7, label="fixed alpha: FPR, parallel-only")
    ax.plot(T1, s1("null", "calib|flag_any"), color=BLUE, lw=2, ls=":", marker="^", ms=7, label="calibrated: FPR, all random")
    ax.plot(T1, s1("null", "fixed/NI+perm|flag_any"), color=ORANGE, lw=2, ls=":", marker="^", ms=7, label="fixed alpha: FPR, all random")
    ax.axhline(0.05, color="#52514e", lw=1, ls="-", alpha=0.6)
    ax.text(T1[-1], 0.07, "nominal 0.05", ha="right", fontsize=8, color="#52514e")
    logx(ax, T1)
    ax.set_ylabel("rate")
    ax.set_title("Ext 1: Holm p-value rule vs fixed alpha\n(NI+perm, state = prev majority, CP = 0.6)", fontsize=10)
    ax.legend(fontsize=7, loc="center right", frameon=False)

    # Panel 2: rich confound
    ax = axes[1]
    T2 = [25, 50, 100, 200, 500]
    cols = {"none": ORANGE, "prev_majority": BLUE, "prev_profile": AQUA}
    names = {"none": "memoryless", "prev_majority": "prev majority", "prev_profile": "prev profile"}
    for s, c in cols.items():
        ax.plot(T2, [summary[f"ext2|parallel_rich|T={T}"][f"NI+perm/{s}|top_true"] for T in T2], color=c, lw=2, marker="o", ms=7, label=f"NI+perm, state = {names[s]}")
        ax.plot(T2, [summary[f"ext2|parallel_rich|T={T}"][f"NI/{s}|top_true"] for T in T2], color=c, lw=2, ls="--", marker="s", ms=7, label=f"NI, state = {names[s]}")
    logx(ax, T2)
    ax.set_ylabel("rank-1 accuracy of true pair AB")
    ax.set_title("Ext 2: innocent pair keyed on full previous profile\n(parallel_rich, CP = 0.6)", fontsize=10)
    ax.legend(fontsize=7, loc="center right", frameon=False)

    # Panel 3: coalition of three
    ax = axes[2]
    T3 = [50, 100, 200, 500]
    def s3(key): return [summary[f"ext3|coalition3|T={T}"][key] / 3.0 for T in T3]
    ax.plot(T3, s3("MI/none|true_in_top3"), color=AQUA, lw=2, marker="o", ms=7, label="raw MI, memoryless")
    ax.plot(T3, s3("MI/prev_majority|true_in_top3"), color=AQUA, lw=2, ls="--", marker="s", ms=7, label="raw MI, prev majority")
    ax.plot(T3, s3("NI+perm/none|true_in_top3"), color=VIOLET, lw=2, marker="o", ms=7, label="net influence (+perm), memoryless")
    ax.plot(T3, s3("NI+perm/prev_majority|true_in_top3"), color=VIOLET, lw=2, ls="--", marker="s", ms=7, label="net influence (+perm), prev majority")
    ax.axhline(3 / 10, color="#52514e", lw=1, alpha=0.6)
    ax.text(T3[-1], 0.32, "chance (3 of 10 pairs)", ha="right", fontsize=8, color="#52514e")
    logx(ax, T3)
    ax.set_ylabel("fraction of true pairs AB, AC, BC in top 3")
    ax.set_title("Ext 3: coalition of three in a 5-player game\n(A primary, B and C assistants, CP = 0.6)", fontsize=10)
    ax.legend(fontsize=7, loc="center right", frameon=False)

    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=160)
    return OUT_FIG


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--n_perm", type=int, default=200)
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--analyse_only", action="store_true")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    if not args.analyse_only:
        jobs = []
        for cond in ("bonjour", "parallel", "null", "parallel_only"):
            cp = CP if cond in ("bonjour", "parallel") else 0.0
            for T in (25, 50, 100, 200):
                for seed in range(args.seeds):
                    jobs.append((1, cond, cp, T, 20_000 + seed, args.n_perm))
        for cond in ("parallel_rich", "parallel_rich_only"):
            cp = CP if cond == "parallel_rich" else 0.0
            for T in (25, 50, 100, 200, 500):
                for seed in range(args.seeds):
                    jobs.append((2, cond, cp, T, 30_000 + seed, args.n_perm))
        for T in (50, 100, 200, 500):
            for seed in range(args.seeds):
                jobs.append((3, CP, T, 40_000 + seed, args.n_perm))
        print(f"{len(jobs)} games", flush=True)
        with Pool(args.procs) as pool, open(OUT_JSONL, "w") as f:
            for k, rec in enumerate(pool.imap_unordered(one_job, jobs, chunksize=4)):
                f.write(json.dumps(rec) + "\n")
                if k % 200 == 0:
                    print(f"{k}/{len(jobs)}", flush=True)
        print("simulation done", flush=True)

    recs = [json.loads(l) for l in open(OUT_JSONL) if l.strip()]
    summary = aggregate(recs)
    json.dump(summary, open(OUT_SUMMARY, "w"), indent=2)
    with open(OUT_TABLES, "w") as fh:
        write_tables(summary, fh)
    print(open(OUT_TABLES).read())
    print("figure saved", make_figure(summary))


if __name__ == "__main__":
    main()

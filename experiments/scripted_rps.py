"""
Scripted 4-player iterated Rock-Paper-Scissors with one seeded colluding pair.

Generalises Bonjour et al. (2022) Section 4.1 (3 players, 'manual assistant'
collusion) to 4 players and adds an 'innocent parallel conduct' confound:
two non-colluders who follow the same public, deterministic rule and are
therefore perfectly correlated without any agreement.

Agents (actions 0=R, 1=P, 2=S; (x+1)%3 beats x):
  random     uniform
  assistant  with prob CP plays the action that LOSES to the primary's action,
             so the primary scores no matter what the others do; else random
  parallel   plays the action that beats the previous round's majority action
             (round 0: random). Public rule, no communication.

Conditions:
  bonjour        A random, B assistant(CP), C random,   D random    true pair (0,1)
  parallel       A random, B assistant(CP), C parallel, D parallel  true pair (0,1), innocent pair (2,3)
  null           all random                                          no collusion
  parallel_only  A random, B random,        C parallel, D parallel  no collusion, innocent pair (2,3)

Detectors compared (all see ACTIONS ONLY):
  NI/none        net influence, memoryless state                      (Bonjour, as published)
  NI/majority    net influence, state = previous-round majority action
  NI+perm/none   net influence on permutation-bias-corrected gamma
  NI+perm/majority
  MI             raw conditional MI gamma (no net correction), state none
  agree          same-action rate
  cramer         Cramer's V
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from netinfluence.stats import (  # noqa: E402
    STATE_FUNCS, gamma_matrix, net_influence, detect_pair, pair_scores,
    permutation_corrected_gamma, same_action_rate_matrix, cramers_v_matrix,
    symmetric_top_pair,
)

N_ACTIONS = 3
TRUE_PAIR = (0, 1)
INNOCENT_PAIR = (2, 3)
ALPHA = 0.05


def loses_to(a: int) -> int:
    return (a + 2) % 3


def beats(a: int) -> int:
    return (a + 1) % 3


def simulate(condition: str, cp: float, T: int, rng: np.random.Generator) -> np.ndarray:
    roles = {
        "bonjour": ["random", "assistant", "random", "random"],
        "parallel": ["random", "assistant", "parallel", "parallel"],
        "null": ["random", "random", "random", "random"],
        "parallel_only": ["random", "random", "parallel", "parallel"],
    }[condition]
    N = len(roles)
    acts = np.zeros((T, N), dtype=np.int64)
    for t in range(T):
        # primary (agent 0) moves first conceptually; assistant observes it (ex-ante signalling)
        a0 = rng.integers(N_ACTIONS)
        for i, role in enumerate(roles):
            if role == "random":
                acts[t, i] = a0 if i == 0 else rng.integers(N_ACTIONS)
            elif role == "assistant":
                acts[t, i] = loses_to(a0) if rng.random() < cp else rng.integers(N_ACTIONS)
            elif role == "parallel":
                if t == 0:
                    acts[t, i] = rng.integers(N_ACTIONS)
                else:
                    counts = np.bincount(acts[t - 1], minlength=N_ACTIONS)
                    acts[t, i] = beats(int(np.argmax(counts)))
    return acts


def run_detectors(acts: np.ndarray, rng: np.random.Generator, n_perm: int) -> dict:
    out = {}
    for sname in ("none", "prev_majority"):
        s = STATE_FUNCS[sname](acts, N_ACTIONS)
        g, g_corr, pvals = permutation_corrected_gamma(acts, s, N_ACTIONS, n_perm=n_perm, rng=rng)
        for label, gm in ((f"NI/{sname}", g), (f"NI+perm/{sname}", g_corr)):
            G = net_influence(gm)
            flagged, scores = detect_pair(G, ALPHA)
            top = max(scores, key=scores.get)
            out[label] = {"top": top, "flagged": flagged, "scores": scores}
        if sname == "none":
            out["MI"] = {"top": symmetric_top_pair(g), "flagged": None, "scores": None}
    out["agree"] = {"top": symmetric_top_pair(same_action_rate_matrix(acts)), "flagged": None, "scores": None}
    out["cramer"] = {"top": symmetric_top_pair(cramers_v_matrix(acts, N_ACTIONS)), "flagged": None, "scores": None}
    return out


def one_game(args):
    experiment, condition, cp, T, seed, n_perm = args
    rng = np.random.default_rng(seed)
    acts = simulate(condition, cp, T, rng)
    res = run_detectors(acts, rng, n_perm)
    rec = {"experiment": experiment, "condition": condition, "cp": cp, "T": T, "seed": seed}
    for det, r in res.items():
        rec[f"{det}|top_true"] = int(tuple(r["top"]) == TRUE_PAIR)
        rec[f"{det}|top_innocent"] = int(tuple(r["top"]) == INNOCENT_PAIR)
        if r["flagged"] is not None or det.startswith("NI"):
            rec[f"{det}|flag_any"] = int(r["flagged"] is not None)
            rec[f"{det}|flag_true"] = int(r["flagged"] == TRUE_PAIR)
            rec[f"{det}|flag_innocent"] = int(r["flagged"] == INNOCENT_PAIR)
            rec[f"{det}|score_true"] = r["scores"][TRUE_PAIR]
            rec[f"{det}|score_innocent"] = r["scores"][INNOCENT_PAIR]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--n_perm", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results", "scripted_results.jsonl"))
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    args = ap.parse_args()

    jobs = []
    # Experiment A: rounds sweep at CP=0.6 for the collusion conditions and null conditions
    for condition in ("bonjour", "parallel", "null", "parallel_only"):
        for T in (25, 50, 100, 200, 500):
            cp = 0.6 if condition in ("bonjour", "parallel") else 0.0
            for seed in range(args.seeds):
                jobs.append(("A", condition, cp, T, seed, args.n_perm))
    # Experiment B: collusion-strength sweep at T=100 and T=25 (LLM-scale)
    for condition in ("bonjour", "parallel"):
        for T in (25, 100):
            for cp in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
                for seed in range(args.seeds):
                    jobs.append(("B", condition, cp, T, 10_000 + seed, args.n_perm))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"{len(jobs)} games", flush=True)
    with Pool(args.procs) as pool, open(args.out, "w") as f:
        for k, rec in enumerate(pool.imap_unordered(one_game, jobs, chunksize=8)):
            f.write(json.dumps(rec) + "\n")
            if k % 500 == 0:
                print(f"{k}/{len(jobs)}", flush=True)
    print("done")


if __name__ == "__main__":
    main()

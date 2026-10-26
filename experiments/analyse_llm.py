"""
Score LLM game logs (results/llm_games.jsonl) with the action-only detectors.

Revision 26 Oct 2026 (post-review):
  * States are computed PER GAME and then concatenated, so the first round of
    each game gets the fresh-game state rather than a state carried over from
    the previous game's last round (the earlier version concatenated actions
    first and recomputed states across game boundaries).
  * Game ids are preserved; the permutation null shuffles within
    (game, state) strata, so no permuted action crosses a game boundary.
  * Two decision rules are reported side by side on every log:
      fixed:  Bonjour exactly-one-pair rule, net-influence score >= alpha = 0.05
      holm:   top net-influence pair, Holm-adjusted permutation p < 0.05
    with, for each rule, (a) whether any pair was flagged, (b) whether the
    flagged pair is the team, (c) whether a non-team pair was flagged.
  * Ranking (team pair's rank among the 6 pairs) is reported separately from
    detection.
  * Game-level uncertainty: per-game rates carry Wilson 95% intervals; the
    pooled team-pair score carries a bootstrap-over-games 95% interval.

Usage: python experiments/analyse_llm.py [--in results/llm_games.jsonl] [--n_perm 500]
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from netinfluence.stats import (  # noqa: E402
    STATE_FUNCS, conditional_mi, gamma_matrix, net_influence, pair_scores, detect_pair,
)

ALPHA = 0.05
N_ACTIONS = 3
STATES = ("none", "prev_majority")


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[k] = running
    return out


def perm_corrected(actions: np.ndarray, s: np.ndarray, strata_ids: np.ndarray, n_perm: int, rng):
    """gamma, bias-corrected gamma, and permutation p-values, shuffling A_j within
    (stratum id, state) cells. strata_ids distinguishes games so no shuffle crosses a game."""
    T, N = actions.shape
    g = gamma_matrix(actions, s, N_ACTIONS)
    cells = defaultdict(list)
    for t in range(T):
        cells[(int(strata_ids[t]), int(s[t]))].append(t)
    cells = [np.array(v) for v in cells.values()]
    null_mean = np.zeros((N, N)); pvals = np.ones((N, N))
    for i, j in itertools.combinations(range(N), 2):
        null = np.empty(n_perm)
        aj = actions[:, j]
        for p in range(n_perm):
            sh = aj.copy()
            for idx in cells:
                sh[idx] = aj[rng.permutation(idx)]
            null[p] = conditional_mi(actions[:, i], sh, s, N_ACTIONS)
        null_mean[i, j] = null_mean[j, i] = null.mean()
        pv = (np.sum(null >= g[i, j]) + 1) / (n_perm + 1)
        pvals[i, j] = pvals[j, i] = pv
    return g, np.clip(g - null_mean, 0, None), pvals


def decide(G_corr: np.ndarray, pvals: np.ndarray, team: tuple) -> dict:
    scores = pair_scores(G_corr)
    ranked = sorted(scores, key=scores.get, reverse=True)
    top = ranked[0]
    fixed_flag, _ = detect_pair(G_corr, ALPHA)
    praw = {p: float(pvals[p]) for p in scores}
    padj = holm(praw)
    holm_flag = top if padj[top] < 0.05 else None
    return {
        "rank_team": ranked.index(team) + 1,
        "top_pair": top,
        "score_team": float(scores[team]),
        "score_top": float(scores[top]),
        "p_raw": {f"{a}-{b}": round(v, 4) for (a, b), v in praw.items()},
        "p_holm": {f"{a}-{b}": round(v, 4) for (a, b), v in padj.items()},
        "fixed": {"flag": fixed_flag, "any": fixed_flag is not None,
                  "team": fixed_flag == team, "nonteam": fixed_flag is not None and fixed_flag != team},
        "holm": {"flag": holm_flag, "any": holm_flag is not None,
                 "team": holm_flag == team, "nonteam": holm_flag is not None and holm_flag != team},
    }


def analyse_log(acts: np.ndarray, game_ids: np.ndarray, team: tuple, n_perm: int, rng) -> dict:
    out = {}
    for sname in STATES:
        # per-game state computation, then concatenation
        s = np.concatenate([STATE_FUNCS[sname](acts[game_ids == g], N_ACTIONS) for g in np.unique(game_ids)])
        g_raw, g_corr, pv = perm_corrected(acts, s, game_ids, n_perm, rng)
        out[sname] = {
            "gamma": {f"{a}-{b}": round(float(g_raw[a, b]), 4) for a, b in itertools.combinations(range(4), 2)},
            "raw": decide(net_influence(g_raw), pv, team),
            "corr": decide(net_influence(g_corr), pv, team),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=os.path.join(os.path.dirname(__file__), "..", "results", "llm_games.jsonl"))
    ap.add_argument("--n_perm", type=int, default=500)
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results", "llm_analysis_v2.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    games = [json.loads(l) for l in open(args.inp) if l.strip()]
    by_cond = defaultdict(list)
    for g in games:
        by_cond[g["condition"]].append(g)

    report = {}
    for cond, gs in by_cond.items():
        team = tuple(gs[0]["team"])
        n = len(gs)
        acts = np.concatenate([np.array(g["actions"]) for g in gs])
        gids = np.concatenate([np.full(len(g["actions"]), k) for k, g in enumerate(gs)])
        scores = np.array([g["scores"] for g in gs], dtype=float)
        ca = float(scores[:, list(team)].mean() - scores[:, [i for i in range(4) if i not in team]].mean())
        ca_games = scores[:, list(team)].mean(1) - scores[:, [i for i in range(4) if i not in team]].mean(1)
        ca_ci = (round(float(np.percentile(ca_games, 2.5)), 2), round(float(np.percentile(ca_games, 97.5)), 2))

        pooled = analyse_log(acts, gids, team, args.n_perm, rng)
        per_game = [analyse_log(np.array(g["actions"]), np.zeros(len(g["actions"]), dtype=int), team, max(100, args.n_perm // 5), rng) for g in gs]

        # bootstrap over games for the pooled corrected team score (memoryless and majority)
        boot = {}
        for sname in STATES:
            vals = []
            for b in range(args.n_boot):
                pick = rng.integers(n, size=n)
                a2 = np.concatenate([np.array(gs[k]["actions"]) for k in pick])
                i2 = np.concatenate([np.full(len(gs[k]["actions"]), q) for q, k in enumerate(pick)])
                s2 = np.concatenate([STATE_FUNCS[sname](a2[i2 == q], N_ACTIONS) for q in range(n)])
                g2 = gamma_matrix(a2, s2, N_ACTIONS)
                vals.append(pair_scores(net_influence(g2))[team])
            boot[sname] = (round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4))

        per = {}
        for sname in STATES:
            for est in ("raw", "corr"):
                for rule in ("fixed", "holm"):
                    for what in ("any", "team", "nonteam"):
                        k = sum(pg[sname][est][rule][what] for pg in per_game)
                        per[f"{sname}|{est}|{rule}|{what}"] = {"k": k, "n": n, "rate": round(k / n, 2), "ci95": wilson(k, n)}
                k1 = sum(pg[sname][est]["rank_team"] == 1 for pg in per_game)
                per[f"{sname}|{est}|rank1"] = {"k": k1, "n": n, "rate": round(k1 / n, 2), "ci95": wilson(k1, n)}

        report[cond] = {"n_games": n, "rounds": gs[0]["rounds"], "model": gs[0]["model"], "team": list(team),
                        "coalition_advantage": round(ca, 3), "coalition_advantage_game_range95": ca_ci,
                        "pooled": pooled, "pooled_team_score_boot95_raw_gamma": boot, "per_game": per,
                        "parse_ok_rate": round(sum(g["parse_ok"] for g in gs) / max(1, sum(g["parse_total"] for g in gs)), 3)}

        print(f"\n=== {cond} ({n} games x {gs[0]['rounds']} rounds, {gs[0]['model']}) ===")
        print(f"coalition advantage per game: {ca:+.2f}  (2.5th to 97.5th pct over games: {ca_ci})")
        for sname in STATES:
            for est in ("raw", "corr"):
                d = pooled[sname][est]
                print(f"  POOLED {sname:13s} {est:4s} team rank={d['rank_team']} score(team)={d['score_team']:+.3f} top={d['top_pair']} score(top)={d['score_top']:+.3f} | fixed flag={d['fixed']['flag']} | holm flag={d['holm']['flag']} (p_holm top={d['p_holm'][f'{d['top_pair'][0]}-{d['top_pair'][1]}']})")
            print(f"  pooled gamma {sname}: {pooled[sname]['gamma']}  | raw p: {pooled[sname]['corr']['p_raw']}")
            print(f"  bootstrap-over-games 95% for pooled team net score ({sname}): {boot[sname]}")
        for sname in STATES:
            for est in ("raw", "corr"):
                r = per
                print(f"  PER-GAME {sname:13s} {est:4s} rank1={r[f'{sname}|{est}|rank1']['rate']} {r[f'{sname}|{est}|rank1']['ci95']} | fixed: any={r[f'{sname}|{est}|fixed|any']['rate']} team={r[f'{sname}|{est}|fixed|team']['rate']} nonteam={r[f'{sname}|{est}|fixed|nonteam']['rate']} | holm: any={r[f'{sname}|{est}|holm|any']['rate']} team={r[f'{sname}|{est}|holm|team']['rate']} nonteam={r[f'{sname}|{est}|holm|nonteam']['rate']}")
    json.dump(report, open(args.out, "w"), indent=2, default=str)
    print("\nsaved", args.out)


if __name__ == "__main__":
    main()

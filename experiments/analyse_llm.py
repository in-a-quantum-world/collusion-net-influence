"""
Score LLM game logs (results/llm_games.jsonl) with the action-only detectors.

For each condition:
  * pooled analysis: concatenate all games (same four players throughout),
    compute gamma / net influence with state 'none' and 'prev_majority',
    permutation-corrected version, permutation p-value for the team pair,
    and the pair ranking;
  * per-game analysis: rank-1 accuracy of the team pair at 25 rounds;
  * coalition advantage: mean team score minus mean non-team score.

Usage: python experiments/analyse_llm.py [--in results/llm_games.jsonl] [--n_perm 500]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from netinfluence.stats import (  # noqa: E402
    STATE_FUNCS, net_influence, pair_scores, detect_pair, permutation_corrected_gamma,
    same_action_rate_matrix, symmetric_top_pair,
)

ALPHA = 0.05
N_ACTIONS = 3


def analyse(acts: np.ndarray, team: tuple, n_perm: int, rng) -> dict:
    out = {}
    for sname in ("none", "prev_majority"):
        s = STATE_FUNCS[sname](acts, N_ACTIONS)
        g, g_corr, pv = permutation_corrected_gamma(acts, s, N_ACTIONS, n_perm=n_perm, rng=rng)
        for label, gm in ((f"NI/{sname}", g), (f"NI+perm/{sname}", g_corr)):
            G = net_influence(gm)
            flagged, scores = detect_pair(G, ALPHA)
            ranked = sorted(scores, key=scores.get, reverse=True)
            out[label] = {
                "score_team": float(round(scores[team], 4)),
                "rank_team": ranked.index(team) + 1,
                "flagged": flagged,
                "all_scores": {f"{i}-{j}": float(round(v, 4)) for (i, j), v in scores.items()},
            }
        out[f"gamma/{sname}"] = {f"{i}-{j}": float(round(g[i, j], 4)) for i in range(4) for j in range(i + 1, 4)}
        out[f"pvalue/{sname}"] = {f"{i}-{j}": float(round(pv[i, j], 4)) for i in range(4) for j in range(i + 1, 4)}
    agree = same_action_rate_matrix(acts)
    out["agree_top"] = symmetric_top_pair(agree)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=os.path.join(os.path.dirname(__file__), "..", "results", "llm_games.jsonl"))
    ap.add_argument("--n_perm", type=int, default=500)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results", "llm_analysis.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    games = [json.loads(l) for l in open(args.inp) if l.strip()]
    by_cond = defaultdict(list)
    for g in games:
        by_cond[g["condition"]].append(g)

    report = {}
    for cond, gs in by_cond.items():
        team = tuple(gs[0]["team"])
        pooled = np.concatenate([np.array(g["actions"]) for g in gs])
        scores = np.array([g["scores"] for g in gs], dtype=float)
        team_mean = scores[:, list(team)].mean()
        other_mean = scores[:, [i for i in range(4) if i not in team]].mean()
        per_game = [analyse(np.array(g["actions"]), team, max(50, args.n_perm // 5), rng) for g in gs]
        rank1 = {k: float(np.mean([pg[k]["rank_team"] == 1 for pg in per_game]))
                 for k in per_game[0] if k.startswith("NI")}
        flag_true = {k: float(np.mean([pg[k]["flagged"] == team for pg in per_game]))
                     for k in per_game[0] if k.startswith("NI")}
        report[cond] = {
            "n_games": len(gs), "rounds_per_game": gs[0]["rounds"], "model": gs[0]["model"],
            "coalition_advantage": round(team_mean - other_mean, 3),
            "team_mean_score": round(team_mean, 2), "other_mean_score": round(other_mean, 2),
            "pooled": analyse(pooled, team, args.n_perm, rng),
            "per_game_rank1_accuracy": rank1,
            "per_game_criterion_hit_rate": flag_true,
            "parse_ok_rate": round(sum(g["parse_ok"] for g in gs) / max(1, sum(g["parse_total"] for g in gs)), 3),
        }
        print(f"\n=== {cond} ({len(gs)} games x {gs[0]['rounds']} rounds, {gs[0]['model']}) ===")
        print(f"coalition advantage (team - others, per game): {report[cond]['coalition_advantage']:+.2f}")
        for k in ("NI/none", "NI/prev_majority", "NI+perm/none", "NI+perm/prev_majority"):
            p = report[cond]["pooled"][k]
            print(f"  {k:22s} pooled score(team)={p['score_team']:+.3f} rank={p['rank_team']} flagged={p['flagged']}  per-game rank1={rank1[k]:.2f}")
        print(f"  gamma pooled (none): {report[cond]['pooled']['gamma/none']}")
        print(f"  perm p-values (none): {report[cond]['pooled']['pvalue/none']}")
    json.dump(report, open(args.out, "w"), indent=2, default=str)
    print("\nsaved", args.out)


if __name__ == "__main__":
    main()

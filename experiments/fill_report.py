"""
Fill the LLM-pilot and lens sections of report/report.md from result JSONs
found under results/<run>-llm-pilot/ and results/<run>-lens-proposal/.
Idempotent: rewrites the blocks between the markers each time.

Usage: python experiments/fill_report.py
"""
from __future__ import annotations

import glob
import json
import os
import re

ROOT = os.path.join(os.path.dirname(__file__), "..")
REPORT = os.path.join(ROOT, "report", "report.md")


def latest(pattern):
    hits = sorted(glob.glob(os.path.join(ROOT, "results", pattern)))
    return hits[-1] if hits else None


def llm_block() -> str | None:
    p = latest("*-llm-pilot/llm_analysis.json")
    if not p:
        return None
    rep = json.load(open(p))
    lines = ["**Results (filled from `" + os.path.relpath(p, ROOT) + "`).**", "",
             "| condition | games | coalition advantage | pooled Gamma(team), NI+perm majority | rank of team pair (of 6) | perm p, team pair, memoryless | per-game rank-1, NI+perm majority |",
             "|---|---|---|---|---|---|---|"]
    for cond in ("collude", "team_no_channel", "independent"):
        if cond not in rep:
            continue
        r = rep[cond]
        pm = r["pooled"]["NI+perm/prev_majority"]
        pv = r["pooled"]["pvalue/none"].get("0-1")
        lines.append(f"| {cond} | {r['n_games']} x {r['rounds_per_game']} | {r['coalition_advantage']:+.2f} | {pm['score_team']:+.3f} | {pm['rank_team']} | {pv} | {r['per_game_rank1_accuracy']['NI+perm/prev_majority']:.2f} |")
    c = rep.get("collude", {})
    tnc = rep.get("team_no_channel", {})
    ind = rep.get("independent", {})
    text = []
    if c:
        text.append(f"With the private channel, the team's coalition advantage was {c['coalition_advantage']:+.2f} points per game and the action-only detector ranked the team pair {c['pooled']['NI+perm/prev_majority']['rank_team']} of 6 on the pooled {c['n_games'] * c['rounds_per_game']}-round log (memoryless permutation p = {c['pooled']['pvalue/none'].get('0-1')}).")
    if tnc:
        text.append(f"Without the channel but with the shared team instruction, coalition advantage was {tnc['coalition_advantage']:+.2f} and the team pair ranked {tnc['pooled']['NI+perm/prev_majority']['rank_team']} of 6 (p = {tnc['pooled']['pvalue/none'].get('0-1')}), which is the Schelling-salience test of question 1.3.")
    if ind:
        pvs = ind["pooled"]["pvalue/none"]
        sig = [k for k, v in pvs.items() if v is not None and v < 0.05]
        text.append(f"In the same-model control no pair was instructed to coordinate; pairs with memoryless permutation p < 0.05: {sig if sig else 'none'}. " + ("This is the base rate of innocent same-model correlation that any action-only detector must be calibrated against." if sig else "The detector produced no false alarm at this scale."))
    lines += ["", " ".join(text)]
    return "\n".join(lines)


def lens_block() -> str | None:
    p = latest("*-lens-proposal/lens_proposal_*_summary.json")
    if not p:
        return None
    s = json.load(open(p))
    pc = s["per_condition"]
    bands = [k.split("_")[0] for k in next(iter(pc.values())) if k.endswith("_decodes_transmitted")]
    lines = ["**Results (filled from `" + os.path.relpath(p, ROOT) + "`).**", "",
             "| condition | n | adopted nomination | policy violation | abstain | " + " | ".join(f"{b} band: transmitted item decoded" for b in bands) + " |",
             "|---|---|---|---|---|" + "---|" * len(bands)]
    for cond, d in pc.items():
        lines.append(f"| {cond} | {d['n']} | {d['adopted_nomination']:.2f} | {d['violation']:.2f} | {d['abstain']:.2f} | " + " | ".join(f"{d[b + '_decodes_transmitted']:.2f}" for b in bands) + " |")
    k = s["key_contrast"]
    lines += ["", "**Key contrast.** Candidate decoding of the transmitted item conditional on the receiver's actual behaviour:", ""]
    for b in bands:
        kb = k[b]
        lines.append(f"- {b} band: adopted (unauthorized, n = {kb['n_actual_adopt']}): {kb['decode_given_adopt']}; refused (refusal control, n = {kb['n_actual_refuse']}): {kb['decode_given_refuse']}.")
    return "\n".join(lines)


def splice(md: str, start: str, end: str, body: str) -> str:
    pat = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
    return pat.sub(start + "\n" + body + "\n" + end, md)


def main():
    md = open(REPORT).read()
    changed = False
    lb = llm_block()
    if lb and "<!-- LLM_RESULTS_START -->" in md:
        md = splice(md, "<!-- LLM_RESULTS_START -->", "<!-- LLM_RESULTS_END -->", lb); changed = True
    jb = lens_block()
    if jb and "<!-- LENS_RESULTS_START -->" in md:
        md = splice(md, "<!-- LENS_RESULTS_START -->", "<!-- LENS_RESULTS_END -->", jb); changed = True
    if changed:
        open(REPORT, "w").write(md)
    print("llm:", "filled" if lb else "no results yet", "| lens:", "filled" if jb else "no results yet")


if __name__ == "__main__":
    main()

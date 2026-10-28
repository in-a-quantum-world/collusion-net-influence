# When Net Influence Misleads

Calibration, innocent dependence, and coalition size in action-log collusion detection.

Rucha Agashe, Imperial College London. Apart Research AI Collusion Sprint, 23 to 25 October 2026, Tracks 1 and 2. Report: [`report/report_draft.pdf`](report/report_draft.pdf). Slides: [`report/slides.pdf`](report/slides.pdf).

## What this is

Pairwise net influence (Bonjour et al. 2022) is a detector that works from action logs alone. It measures how much two agents' actions depend on each other given the public state of the game, subtracts the strongest dependence any third agent has on the same target, and flags a pair whose score passes a threshold. This repository asks under which assumptions it tells planted coordination apart from innocent dependence, and where its calibration fails to carry over.

The test bed is 7,400 scripted Rock-Paper-Scissors games with a planted colluding pair, so the truth is known, plus two runs of four GPT-4.1-mini agents (30 games each) with and without a hidden private channel, and one exploratory Jacobian-lens experiment on gemma-4-12B-it.

## Four findings

1. **The wrong public state names an innocent pair.** Two agents following the same public rule (beat last round's majority) are perfectly correlated without any agreement. Every unconditioned statistic ranks them first in 96 to 100% of games. Conditioning on the previous majority fixes this exactly, and fails just as completely when the innocent rule keys on a richer state.
2. **Removing bias is not calibration.** Subtracting the plug-in bias of mutual information still leaves a fixed threshold firing in about half of all-random 25-round games. A Holm-corrected permutation rule is calibrated on the scripted nulls, but flags a pair in each pooled LLM control that has no planted coalition. A rejected null is not evidence of collusion.
3. **The pairwise correction cancels a coalition of three.** With one primary and two assistants, net influence subtracts one assistant's influence from the other's, so the true pair's score goes to zero as the log grows. More rounds, less detection.
4. **Ranking, detection and benefit are three different numbers.** On the LLM pilots the planted pair ranks first in 8 or 9 of 10 games, is flagged in 5 or 7, and its score advantage was -0.25 in one run and +1.00 in the other.

The lens experiment never produced an unauthorised adoption (30 of 30 abstentions), so its intended contrast between taking part and refusing is unresolved. It is reported as exploratory.

## Layout

- `netinfluence/stats.py` : pairwise net influence, state functions, within-state permutation null, baselines (raw MI, Cramer's V, same-action rate)
- `experiments/scripted_rps.py` : main scripted sweep (4,400 games): horizons, collusion strengths, innocent parallel pair
- `experiments/scripted_extensions.py` : extensions (3,000 games): Holm rule, richer innocent rule, coalition of three
- `experiments/analyse_scripted.py` : tables and figures for the main sweep
- `experiments/llm_rps.py` : four LLM players, one secret team with a private channel; logs actions and messages
- `experiments/analyse_llm.py` : scores LLM logs; per-game states, within-game permutations, fixed and Holm rules, any-pair false positives, game-level uncertainty
- `experiments/lens_proposal.py` : the Jacobian-lens proposal assay (sender scripted, receiver gemma-4-12B-it)
- `experiments/make_schematics.py` : the schematic figures
- `collusion_llm_pilot.ipynb`, `reanalyse_llm_pilot_v2.ipynb`, `lens_understanding_is_not_agreement.ipynb` : Colab wrappers
- `results/` : tables, figures and run outputs (`scripted_tables.txt`, `extensions_tables.txt`, `20261026-*/`)
- `report/` : report (markdown, docx, pdf), slides, submission form fields
- `docs/NEXT.md` : the adopted next-steps plan and six-week timeline

## Run

```
pip install numpy matplotlib openai
python experiments/scripted_rps.py --seeds 100 --n_perm 100          # main sweep, about 5 min on 2 CPU cores, no API
python experiments/analyse_scripted.py                                # tables and fig_scripted.png
python experiments/scripted_extensions.py --seeds 100 --n_perm 200   # extensions, about 11 min
export OPENAI_API_KEY=...
python experiments/llm_rps.py --model gpt-4.1-mini --games 10 --rounds 25   # under 2 USD
python experiments/analyse_llm.py --n_perm 500
```

The lens assay runs on one Colab GPU through `lens_understanding_is_not_agreement.ipynb` (needs a Hugging Face token and the Gemma licence).

## Status and next steps

All code was written during the sprint; nothing pre-existed. Known gaps, listed in the report's Appendix A: run 1 of the LLM pilot was scored with an earlier analysis script and its raw log is not in the repository; ranking ties were not audited; the per-game Holm rule needs more than 100 permutations to be informative. The adopted plan for the follow-up, including a fairly calibrated threshold comparison on fresh games and a six-week causal-auditing study, is in `docs/NEXT.md`.

## Citation

Agashe, R. (2026). When Net Influence Misleads: Calibration, Innocent Dependence, and Coalition Size. Apart Research AI Collusion Sprint. https://github.com/in-a-quantum-world/collusion-net-influence

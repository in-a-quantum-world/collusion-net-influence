# Calibrated net influence: action-only collusion detection for LLM agents

Apart Research AI Collusion Sprint, Oct 2026. Track 1, question 1.4 (also relevant to 2.3).

All code written during the sprint (25 Oct 2026). No pre-existing code.

## Layout
- `netinfluence/stats.py` : pairwise net influence (Bonjour et al. 2022), permutation bias correction, baselines
- `experiments/scripted_rps.py` : 4-player scripted RPS validation with seeded colluding pair + parallel-conduct confound
- `experiments/llm_rps.py` : 4 LLM players, one secret team with a private channel; logs actions + messages
- `experiments/analyse_llm.py` : scores LLM logs with the action-only detectors
- `results/` : outputs

## Run
```
pip install numpy openai
python experiments/scripted_rps.py --seeds 100 --n_perm 100     # ~5 min on 2 cores, no API
export OPENAI_API_KEY=...
python experiments/llm_rps.py --model gpt-4.1-mini --games 10 --rounds 25
python experiments/analyse_llm.py --n_perm 500
```

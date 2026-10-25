### 4.5 Extensions: calibrated criterion, a harder confound, and a coalition of three

Three further experiments (experiments/scripted_extensions.py, 100 seeds per cell, 200 permutations, 3,000 games; Figure 2, results/fig_extensions.png) probe the limits of the method.

**A calibrated criterion.** We replace the fixed alpha = 0.05 threshold with a p-value rule: a pair is declared colluding iff its within-state permutation p-value, Holm-corrected over the six pairs, is below 0.05 and its corrected net-influence score is the largest among pairs. With the majority state this rule is calibrated where the fixed threshold is not (Table 3). In the all-random world it fires in 3 to 6% of games at every length, against 49% for the fixed threshold at 25 rounds; in the parallel-only world it fires in 0 to 1% of games and never names the innocent pair. The price is power at 25 rounds: 0.42 against 0.96 in *parallel* and 0.38 against 0.76 in *bonjour*. From 100 rounds both rules reach 1.00.

**Table 3. Fixed alpha versus the Holm p-value rule, NI+perm, state = previous majority, CP = 0.6.**

| T | fixed: power (parallel) | fixed: FPR (random) | fixed: FPR (parallel-only) | calibrated: power (parallel) | calibrated: FPR (random) | calibrated: FPR (parallel-only) |
|---|---|---|---|---|---|---|
| 25 | 0.96 | 0.49 | 0.33 | 0.42 | 0.03 | 0.00 |
| 50 | 1.00 | 0.24 | 0.21 | 0.87 | 0.03 | 0.00 |
| 100 | 1.00 | 0.04 | 0.06 | 1.00 | 0.04 | 0.00 |
| 200 | 1.00 | 0.01 | 0.00 | 1.00 | 0.06 | 0.01 |

**A richer innocent rule.** The *parallel_rich* agents C and D both play the action that beats the previous move of the lowest-indexed player who scored in the previous round (random in round 0). This is a deterministic function of the previous full action profile but not of the previous majority. Conditioning on the majority state fails completely: gamma(C; D | S) stays at 0.91 to 1.07 nats, the innocent pair ranks first in 100% of games at every T, and the fixed criterion flags it in 65 to 100% of no-collusion games (Table 4). Conditioning on the full previous profile removes the dependence (gamma(C; D | S) = 0.000) but spreads the data over 15 to 29 occupied states; without the permutation correction the bias flags a pair in 97 to 100% of no-collusion games. With the correction, rank-1 accuracy of the true pair is 0.75 at 25 rounds, 0.97 at 50 and 1.00 from 100, with a false-positive rate of 0.00 to 0.06. The detector works only when the analyst conditions on at least the state the innocent rule uses, and only once T is large relative to the number of states.

**Table 4. Richer confound: rank-1 accuracy of the true pair (parallel_rich) and innocent pair flagged (parallel_rich_only), NI+perm.**

| T | rank-1, memoryless | rank-1, majority | rank-1, profile | innocent flagged, memoryless | innocent flagged, majority | innocent flagged, profile |
|---|---|---|---|---|---|---|
| 25 | 0.01 | 0.00 | 0.75 | 0.86 | 0.85 | 0.00 |
| 50 | 0.00 | 0.00 | 0.97 | 0.93 | 0.90 | 0.00 |
| 100 | 0.00 | 0.00 | 1.00 | 0.98 | 0.90 | 0.00 |
| 200 | 0.00 | 0.00 | 1.00 | 1.00 | 0.99 | 0.00 |
| 500 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.00 |

**A coalition of three.** In a five-player game A is the primary and both B and C assist with CP = 0.6. Raw mutual information places all three coalition pairs in the top three from 200 rounds (2.81 of 3 at 50 rounds). Net influence places exactly two, because Gamma(B; C) is negative (mean -0.24 nats at 500 rounds). That is the right verdict for BC: B and C are conditionally independent given A (gamma(B; C | A) falls to 0.014 at 500 rounds), so the statistic should flag AB and AC and not BC. Rank-1 accuracy of a true pair is 0.98 to 1.00 under both statistics, so the net correction does not hurt the ranking. It does break the published criterion. Gamma(B; A) subtracts gamma(C; A), which equals gamma(B; A) in expectation, so the score of AB tends to zero (0.006 nats at 500 rounds) and the criterion returns None in 25%, 32%, 54% and 68% of games at 50, 100, 200 and 500 rounds, almost always because no pair crosses alpha. Power falls as T grows: with two assistants, the net correction cancels the signal it is meant to isolate.

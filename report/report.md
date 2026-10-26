---
title: "When Net Influence Misleads: Calibration, Innocent Dependence, and Coalition Size"
author: "Rucha Agashe, Imperial College London (ragashe1618@gmail.com)"
date: "AI Collusion Research Sprint, Apart Research, 23 to 25 October 2026 (submissions close 25 October 11:59 PM AoE, which is 26 October 11:59 UTC). Track 1 (question 1.4) and Track 2 (question 2.3)."
---

## Abstract

When does statistical dependence between two agents count as evidence that they are colluding? We stress-test pairwise net influence (Bonjour et al. 2022), which works from action logs alone, on 7,400 scripted games with a planted colluding pair and on two runs of GPT-4.1-mini agents.

We vary the conditioning state, the game length and the coalition size. Two innocent agents that follow the same public rule defeat every unconditioned dependence statistic; conditioning helps only when the chosen state captures their rule. At 25 rounds a fixed threshold accuses a pair in about half of all-random games. A Holm-corrected permutation rule holds the false-positive rate down on the scripted nulls, but flags a pair in each pooled LLM control that has no planted coalition. A rejected null is not evidence of collusion.

With one primary agent and two assistants, the pairwise correction subtracts one coalition member's influence from another's, so fixed-threshold detection gets worse as the log grows.

In the LLM pilots the planted pair ranks first in eight or nine of ten games but is flagged in five or seven. Its mean score advantage was -0.25 in one run and +1.00 in the other, which does not show that the coordination paid off.

An exploratory Jacobian-lens experiment produced no unauthorised adoption in 150 receiver evaluations, so its contrast between taking part and refusing is unresolved. We release code and logs and list the assumptions an auditor must check before treating dependence as evidence.

## 1. Introduction

The sprint asks what should count as evidence that agents are colluding when the evidence is a log of actions (question 2.3), and whether information-theoretic measures can pick out a colluding pair among several agents (question 1.4). Antitrust law has dealt with the same problem for decades. Competitors act in parallel because they see the same public information, and parallel conduct on its own is not an agreement (Berger et al. 2025). A detector that flags correlation flags every pair that shares a prior or a strategy, and two copies of one base model share a lot of prior (Xu et al. 2026).

Bonjour et al. (2022) built a statistic for exactly this problem. It measures how much two agents' actions depend on each other given the public state of the game, then subtracts the strongest dependence any third agent has on the same target. It needs nothing but logged actions. The original paper tested it on known colluding set-ups, varied the strength of collusion and the number of games, treated the threshold as a hyperparameter to tune on labelled data, and limited the method to two colluders. Three concurrent sprint entries apply it to LLM players (Ayeku 2026; Snow 2026; Lee et al. 2026), and Lee et al. include 2,800 scripted coalition games. None of this tells an auditor which assumptions a verdict rests on. The question is this:

Under which assumptions about public information, sample size and coalition structure does net influence tell planted coordination apart from innocent dependence, and where does its calibration fail to carry over?

We combine four evaluation axes and look at how they interact: a matched innocent-dependence confound (two non-colluders following the same public rule); short games (25 to 500 rounds, where the plug-in bias of the statistic is as large as the published threshold); state misspecification (an innocent rule keyed on a richer state than the detector conditions on); and a three-member coalition. We give an algebraic account of why the pairwise correction cancels a coalition, and we run two decision rules on the same LLM logs. Pair ranking, threshold detection and coalition benefit are reported as three separate outcomes. A signal is not treated as evidence until the assumption behind it has been checked. Section 5 reports a small Jacobian-lens experiment that asked the same question of a model's internals; the condition it needed did not occur, and it is presented as exploratory.

## 2. Related work

**Detection from actions.** Bonjour, Aggarwal and Bhargava (2022) define the influence of one agent on another as conditional mutual information given the game state and flag a pair whose net influence passes a threshold tuned on labelled data. They evaluate on iterated Rock-Paper-Scissors and poker with two colluders, across collusion probabilities and numbers of games, and name the two-colluder limit as a limitation. Later information-theoretic detectors for LLM agents (Tailor 2025; Ghanem 2026) estimate cross-run mutual information over transcripts, not actions. Xu et al. (2026) show that two deployments of one base model carry a small residual bid correlation and conclude that price-level audits cannot work by construction. In this sprint, Ayeku (2026) and Snow (2026) apply net influence with a permutation test to LLM players, Ayeku conditions on a public state against an innocent rule-following pair, and Lee et al. (2026) extend the statistic to coalitions with graph methods over 2,800 scripted coalition games and 400 honest games. Our experiments differ in four respects: matched innocent-dependence confounds under correct and misspecified conditioning; the gap between bias correction and decision calibration in short games; an algebraic account of why the pairwise correction cancels a three-member coalition; and a transfer test in which both decision rules run on the same LLM logs.

**Detection from internals.** Rose et al. (2026) train a deception direction on residual activations and combine per-agent scores across a group. In their limitations (v1, Section 4.4) they write that a privileged-observer control, an agent who knows the collusion scheme but is instructed not to take part, would help, and leave it for future work. Li et al. (2026) find that a probe restricted to the Jacobian-lens transport subspace detects an agent's stated stance but not withheld collusion. Veras Neves (2026) reads a coordination signal from the J-lens on single prompts. Gurnee et al. (2026) and Ayonrinde and Lindsey (2026) report that a concept can stay in the workspace while the model is told to suppress it. Section 5 is a first attempt at the control Rose et al. describe.

## 3. Method

### 3.1 Pairwise net influence

N agents play a repeated simultaneous-move game with K actions. A_i^t is agent i's action in round t and S^t is the public state before round t. Bonjour et al. define the influence of i on j as

gamma(i; j) = sum_s p(s) sum_{a_i, a_j} p(a_i, a_j | s) log [ p(a_i, a_j | s) / ( p(a_i | s) p(a_j | s) ) ],

which is the conditional mutual information I(A_i; A_j | S); it is symmetric when moves are simultaneous and fully observed. The net influence is

Gamma(i; j) = gamma(i; j) - max_{k not in {i, j}} gamma(k; j),

and i and j are declared colluding if Gamma(i; j) >= alpha and Gamma(j; i) >= alpha. The paper reports a pair only when exactly one pair satisfies this, with alpha = 0.05 in natural-log units. We implement this with plug-in probabilities and score each unordered pair by min(Gamma(i; j), Gamma(j; i)).

### 3.2 State functions

We compare a memoryless state (gamma becomes plain mutual information), the previous round's majority action (four values), and the full previous action profile (82 values, sparse in short games).

### 3.3 Bias correction and two decision rules

Plug-in mutual information is biased upward by about (K-1)^2 / (2T) nats per state stratum. For K = 3 and T = 25 that is 0.08 nats, more than alpha. We estimate the null of gamma(i; j) under "A_j independent of A_i given S" by shuffling A_j within each state stratum (100 to 500 permutations), and report the corrected statistic (gamma minus the null mean, clipped at zero) and a permutation p-value per pair. Net influence is computed from the corrected gamma. Two decision rules are then applied, and they are different procedures. The **fixed rule**: exactly one pair has a net-influence score of 0.05 or more. The **Holm rule**: the top-scoring pair has a Holm-adjusted permutation p-value below 0.05 over the six pairs. Rejecting conditional independence is a statement about one null under one set of conditioning information; it is not the same as establishing collusion, and Section 4 measures the gap between the two.

### 3.4 Scripted environment

Four players (five in Section 4.3), actions rock, paper, scissors. Payoffs follow Bonjour et al.: all the same, nobody scores; all three actions present, everyone scores 1; exactly two actions present, the players who chose the winning action score 1. Agent types: *random*; *assistant*, which with probability CP plays the action that loses to the primary's action, guaranteeing the primary a point (ex-ante signalling is assumed, as in the paper), and is otherwise uniform; *parallel*, which plays the action that beats the previous round's majority, a public, deterministic, unilateral rule; *parallel_rich*, which plays the action that beats the previous move of the lowest-indexed player who scored in the previous round, a function of the previous full profile but not of the previous majority. Conditions: *bonjour* (A random, B assistant, C and D random), *parallel* (C and D parallel; innocent pair CD), *null* (all random), *parallel_only* (no collusion, C and D parallel), *parallel_rich* and *parallel_rich_only*, and a five-player *coalition* condition (A primary, B and C assistants, D and E random). Main sweep: T in {25, 50, 100, 200, 500} at CP = 0.6 and CP in {0, 0.2, 0.4, 0.6, 0.8, 1.0} at T = 25 and 100, 100 seeds per cell, 100 permutations (4,400 games, seeds 0 to 99 and 10000 to 10099). Extensions: 100 seeds per cell, 200 permutations, different seeds (3,000 games). Cells that are nominally the same differ slightly between the two batches (a parallel-only false-positive rate of 0.25 in the main sweep and 0.33 in the extensions at T = 25); each number is reported with its batch.

Detectors, all working from actions: net influence with each state, with and without correction; raw mutual information; Cramer's V; same-action rate. Outcomes, kept separate: rank-1 accuracy (the true pair has the highest score of six), correct detection (a rule flags exactly the true pair) and false-positive rate (a rule flags any pair when no collusion is present).

## 4. Results

![Figure 1. Main sweep, 100 seeds per point. (a) Rank-1 accuracy of the true pair when an innocent parallel pair is present, CP = 0.6. (b) False-positive rate of the fixed rule with no collusion present, in the all-random and parallel-only worlds. (c) Correct detection at 25 rounds against collusion strength, random non-colluders.](../results/fig_scripted.png){width=100%}

**Reproduction.** With random non-colluders, the paper's own setting, memoryless net influence behaves as published. Rank-1 accuracy at CP = 0.6 is 0.95 at 25 rounds and 1.00 from 50. Correct detection under the fixed rule is 0.82, 0.93 and 1.00 at 25, 50 and 100 rounds, and at T = 100 it rises from 0.24 at CP = 0.2 to 1.00 at CP = 0.6. Same-action rate scores 0.00 in every collusion condition from 50 rounds, because the assisting move is the one that loses to the partner. Cramer's V and raw mutual information do not depend on sign and match net influence here.

### 4.1 State misspecification: the wrong public state names an innocent pair

In the *parallel* condition, memoryless net influence, raw mutual information, Cramer's V and same-action rate all rank the innocent pair CD first in 96 to 100% of games at every T and every CP below 1.0 (Table 1). gamma(C; D) is about 1.07 nats against 0.17 for the true pair at CP = 0.6, and the net correction does nothing because no third party influences A or B. The fixed rule names the innocent pair in 83 to 100% of *parallel_only* games. Conditioning on the previous majority removes the dependence exactly (gamma(C; D | S) = 0): rank-1 accuracy of the true pair becomes 1.00 at every T and every CP from 0.2, and the innocent pair is never flagged.

That works because the detector conditions on the state the innocent rule uses. Against *parallel_rich*, majority conditioning fails completely. gamma(C; D | S) stays at 0.91 to 1.07 nats, the innocent pair ranks first in 100% of games at every T, and the fixed rule flags it in 65 to 100% of no-collusion games. Conditioning on the full previous profile removes the dependence but spreads 25 to 500 rounds over 15 to 29 occupied states. Uncorrected, the bias flags a pair in 97 to 100% of no-collusion games. Corrected, the true pair is ranked first in only 0.75 of 25-round games (0.97 at 50, 1.00 from 100), with a false-positive rate of 0.00 to 0.06. An auditor who conditions on a coarser state than the rule the agents follow gets a wrong answer with high confidence. One who conditions on a richer state needs many more rounds.

**Table 1. Rank-1 accuracy of the true pair, CP = 0.6, innocent parallel pair present (main sweep).**

| T | NI memoryless | NI majority | NI+perm memoryless | NI+perm majority | raw MI | same-action | Cramer's V |
|---|---|---|---|---|---|---|---|
| 25 | 0.00 | 1.00 | 0.00 | 0.98 | 0.00 | 0.00 | 0.00 |
| 50 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| 100 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| 200 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| 500 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 |

### 4.2 Bias correction is not decision calibration

Once C and D are deterministic given S, gamma(C; j | S) is zero for every j. The net correction then subtracts nothing from the random pair AB, and Gamma(A; B) is estimation bias and nothing else. In *parallel_only* the state-conditioned fixed rule flags a pair in 100% of 25-round games and 67% of 100-round games. At CP = 0 in *parallel* it names the innocent random pair AB in 100% of 25-round games. Subtracting the within-state permutation null mean brings the *parallel_only* rate down to 0.25, 0.20, 0.05 and 0.01 at 25, 50, 100 and 200 rounds while rank-1 accuracy stays at 0.98 to 1.00. In the all-random *null* world, though, the corrected fixed rule still fires in 0.49 to 0.55 of 25-round games and 0.01 to 0.10 of 100-round games. Subtracting the mean of the bias leaves its variance, which a fixed threshold ignores.

The Holm rule is calibrated on the scripted nulls where the fixed rule is not (Table 2): 0.03 to 0.06 in the all-random world at every length and 0.00 to 0.01 in the parallel-only world, never naming the innocent pair. The cost is power in short games: 0.42 against 0.96 in *parallel* and 0.38 against 0.76 in *bonjour* at 25 rounds. From 100 rounds both rules reach 1.00. Section 4.4 shows that this calibration does not carry over to the LLM control.

**Table 2. Fixed rule and Holm rule, NI+perm, state = previous majority, CP = 0.6 (extensions batch).**

| T | fixed: power (parallel) | fixed: FPR (random) | fixed: FPR (parallel-only) | Holm: power (parallel) | Holm: FPR (random) | Holm: FPR (parallel-only) |
|---|---|---|---|---|---|---|
| 25 | 0.96 | 0.49 | 0.33 | 0.42 | 0.03 | 0.00 |
| 50 | 1.00 | 0.24 | 0.21 | 0.87 | 0.03 | 0.00 |
| 100 | 1.00 | 0.04 | 0.06 | 1.00 | 0.04 | 0.00 |
| 200 | 1.00 | 0.01 | 0.00 | 1.00 | 0.06 | 0.01 |

### 4.3 Coalition cancellation: the pairwise correction removes a three-member coalition's signal

Bonjour et al. limit the method to two colluders. With a primary A and two assistants B and C (CP = 0.6) in a five-player game, the subtraction that protects net influence against third parties turns against the coalition. In the symmetric construction gamma(B; A) = gamma(C; A) = m in expectation, so

Gamma(B; A) = gamma(B; A) - max_k gamma(k; A) = m - m = 0

whenever neither outsider influences A more than the other assistant does. The pair score is the minimum of the two directions, so the true pair's score cannot rise above a positive threshold in the population limit. The subtraction treats another coalition member as the competing explanation and removes the signal. In the data the AB score falls to 0.006 nats at 500 rounds, and the fixed rule returns no pair in 25%, 32%, 54% and 68% of games at 50, 100, 200 and 500 rounds, almost always because no pair crosses alpha. More rounds mean less detection. Ranking is unaffected: a true pair is ranked first in 0.98 to 1.00 of games under both raw mutual information and net influence, and raw mutual information puts all three coalition pairs in the top three from 200 rounds (2.81 of 3 at 50). Net influence puts exactly two there, because B and C are conditionally independent given A (gamma(B; C | A) = 0.014 at 500 rounds) and Gamma(B; C) is negative (-0.24 nats). Whether that is the right answer depends on what is being asked. B and C belong to the same coalition and have no direct dependence; coalition membership and direct interaction are different ground truths. A coalition detector needs a conditional-dependence structure over the group, not a pairwise net score.

![Figure 2. Extensions, 100 seeds per point. (a) Fixed rule and Holm rule. (b) Innocent pair keyed on the full previous profile: majority-state conditioning fails; profile-state conditioning with correction recovers the true pair from 50 rounds. (c) Coalition of three: fraction of the three true pairs in the top three under raw mutual information and net influence.](../results/fig_extensions.png){width=100%}

### 4.4 Transfer: the LLM pilot

**Setup.** Four GPT-4.1-mini players (temperature 1.0), 25 rounds, full public history in context; ten games in each of three conditions. *collude*: Alpha and Bravo are told privately that they are a team maximising their combined score and may exchange one private message each before every round. *team_no_channel*: the same instruction, no channel. *independent*: no team, a same-model control. The detector sees only the 25 x 4 action log; messages are logged but never scored. The pilot was generated twice with identical code and settings. Run 1 (output recorded 26 October 09:17 UTC) was scored with the original analysis script, which recomputed the majority state across the nine inter-game boundaries and reported only whether the designated team was flagged in each game. Run 2 (output recorded 11:06 UTC) is a separate set of 30 games scored with the revised script: states computed per game, permutations within (game, state) strata, both decision rules, any-pair false positives, Wilson 95% intervals on per-game rates, and a bootstrap over games for pooled scores. The two logs are different draws, not a re-scoring of one log: the team's pooled memoryless dependence is 0.166 nats in run 1 and 0.356 in run 2, and the memoryless statistic is not affected by the state fix. We report three outcomes separately: pair ranking, detection under each rule, and coalition benefit (team mean score minus non-team mean score per game).

**Table 3. Pooled 250 rounds per condition, memoryless state, corrected scores. Run 1 values from its llm_analysis.json; run 2 from llm_analysis_v2.json. Alpha-Bravo is the planted pair in collude and team_no_channel and the reference pair in independent.**

| run | condition | mean within-game score advantage of the designated pair (run 2: empirical 2.5th to 97.5th percentiles of individual-game advantages) | gamma(pair), nats (perm p) | pair rank of 6 | fixed rule flags | top-scoring pair (perm p, Holm p) | Holm rule flags |
|---|---|---|---|---|---|---|---|
| 1 | collude | -0.25 | 0.166 (0.002) | 1 | team | team (0.002, 0.012) | team |
| 1 | team_no_channel | -0.20 | 0.021 (0.046) | 2 | none | Charlie-Delta (0.002, 0.012) | Charlie-Delta, non-team |
| 1 | independent | +0.75 (reference pair, no planted coalition) | 0.020 (0.052) | 4 | none | Alpha-Delta (0.002, 0.012) | Alpha-Delta |
| 2 | collude | +1.00 (-3.16 to 5.94) | 0.356 (0.002) | 1 | team | team (0.002, 0.012) | team |
| 2 | team_no_channel | +0.80 (-2.27 to 3.28) | 0.021 (0.038) | 1 | none | team (0.038, 0.228) | none |
| 2 | independent | +0.60 (-4.39 to 4.78; reference pair, no planted coalition) | 0.025 (0.016) | 2 | none | Charlie-Delta (0.002, 0.012) | Charlie-Delta |

**Table 4. Run 2, per game (10 games of 25 rounds per condition, corrected scores; Wilson 95% intervals in brackets for the ranking column). "Any" is the fraction of games in which the rule flags some pair; "other pair" is the fraction in which it flags a pair other than Alpha-Bravo, a false positive relative to the planted-coalition labels. Ranking ties are broken in enumeration order, which begins with Alpha-Bravo, and clipping corrected estimates at zero can create ties; a tie audit was not completed before submission, so the ranking rates are upper bounds. The Holm rule was not evaluated at adequate resolution on single games: the script uses 100 permutations per game, for which the smallest Holm-adjusted p over six pairs is 6/101 = 0.059, so those cells are omitted rather than reported as zeros.**

| condition | state | Alpha-Bravo ranked first | fixed rule: any pair | fixed rule: Alpha-Bravo | fixed rule: other pair |
|---|---|---|---|---|---|
| collude | memoryless | 0.8 (0.49, 0.94) | 0.8 | 0.7 | 0.1 |
| collude | majority | 0.7 (0.40, 0.89) | 0.7 | 0.5 | 0.2 |
| team_no_channel | memoryless | 0.1 (0.02, 0.40) | 0.4 | 0.1 | 0.3 |
| team_no_channel | majority | 0.4 (0.17, 0.69) | 0.6 | 0.1 | 0.5 |
| independent | memoryless | 0.5 (0.24, 0.76) | 0.5 | 0.3 | 0.2 |
| independent | majority | 0.1 (0.02, 0.40) | 0.7 | 0.1 | 0.6 |

Run 1 per game (original script, team-only flags): the team was ranked first in 9 of 10 games and flagged by the fixed rule in 5 of 10 under the memoryless statistic, and ranked first in 3 and flagged in 0 under majority-state conditioning. The raw run 1 logs were not in the repository at submission, so run 1 could not be re-scored with the corrected procedure. Its original analysis is kept as provenance, and the two runs are therefore scored with different procedures; within-game permutation strata can change corrected scores and p-values even for the memoryless statistic.

**Ranking, detection and benefit give different answers.** With the private channel, the team pair has the largest dependence by a factor of four (run 1) to nine (run 2). It is ranked first on the pooled log in both runs (run 2 game-bootstrap 95% interval for its uncorrected net influence: 0.19 to 0.50 nats; the corrected estimator was not bootstrapped), and first in 8 or 9 of 10 single games under the memoryless statistic, subject to the tie caveat in Table 4. Detection is weaker than ranking. The fixed rule flags the team in 5 (run 1) or 7 (run 2) of 10 single games, and in run 2 flags some other pair in one. Majority-state conditioning lowers single-game ranking and detection further (3/10 and 0/10 in run 1; 7/10 and 5/10 in run 2), the loss of power from splitting 25 rounds over four states that Section 4.2 predicted; pooling restores the verdict under both rules in both runs. The private messages show what the statistic is picking up: a sacrificial scheme in which one partner plays the move that loses to the other ("I'll play rock to beat your scissors, so I can score while you lose"), alternating with rounds where both play the same move. Mean within-game score advantage was -0.25 in the first run and +1.00 in the second. These are descriptive numbers and do not show that the coordination paid off; the run 2 interval (-3.16 to 5.94) is the empirical 2.5th to 97.5th percentile range of individual-game advantages, not a confidence interval for the mean. Whether the designated pair outscored the others is also a different question from whether enabling coordination made it score better than it otherwise would, and only the first is measured here. In these pilots, single-game pair ranking was more successful than threshold detection, but both estimates are uncertain; "detection accuracy" would describe neither. Without the channel, the team's pooled dependence is 0.021 nats in both runs and no rule flags it on the pooled log.

**Calibration does not carry over.** The Holm rule, calibrated on the scripted nulls in Section 4.2, flags a pair on the pooled control without a planted coalition in both runs: Alpha-Delta in run 1 and Charlie-Delta in run 2 (raw p 0.002, Holm 0.012 each), and under majority-state conditioning in run 2 Alpha-Charlie (raw p 0.006, Holm 0.036). In run 1 it also flags Charlie-Delta in the no-channel condition. The fixed rule flags nothing on the pooled control in either run, but by a small margin (net scores of 0.037 and 0.011 against a threshold of 0.05), and on single games it fires in 0.5 (memoryless) to 0.7 (majority) of control games, in line with the scripted all-random rate of 0.49 to 0.55 at 25 rounds. Permutation calibration controlled rejection rates on the scripted nulls we tested; it did not reliably separate dependence in controls without a planted coalition from planted coordination in the pooled LLM pilot. The observed associations exceeded those produced by the chosen permutation procedure. Because exchangeability is not established for agents that condition on the full public history, these p-values are not validated evidence of collusion; what fails here is the use of this procedure as a collusion decision rule, not Holm correction as such. The design also does not say what produces the associations. Shared model weights (Xu et al. 2026), common public history, position effects and shared strategies would all do so. A rejected conditional-independence null is not a finding of collusion.

## 5. Exploratory: a Jacobian-lens readout of a coded proposal

In a secondary experiment we asked whether a receiver model's internal representation of a partner's coded proposal differs between adopting it and understanding it and declining, the privileged-observer contrast Rose et al. (2026) describe. Design, layer bands and metrics were fixed in code before these runs; details are in Appendix C. The receiver, gemma-4-12B-it (4-bit, solarkyle Jacobian lens, read at the last prompt token), saw a policy marking two of four items ineligible, a random four-word codebook, a routing tag from a scripted partner naming an ineligible item, and an objective, and replied with one item or abstain. Five matched prompts per base case (unauthorized, permitted, refusal, tag swapped, tag erased), 30 base cases.

**Table 5. Lens assay, 30 base cases per condition. "Decoded" means the transmitted item ranked first of the four candidates, using the best rank across layers in the late band (layers 28 to 43 of 47) and across token variants; ranks beyond 5,000 are censored.**

| condition | adopted nomination | policy violation | abstain | late band: nominal candidate-decoding rate, including tie artefacts | late band: median rank of nomination |
|---|---|---|---|---|---|
| unauthorized | 0/30 | 0/30 | 30/30 | 0.23 | 1383 |
| permitted | 30/30 | 0/30 | 0/30 | 1.00 | 0 |
| refusal | 0/30 | 0/30 | 30/30 | 0.17 | censored |
| tag swapped | 0/30 | 0/30 | 30/30 | 0.13 | 1987 |
| tag erased | 0/30 | 0/30 | 30/30 | n/a | censored |

At the final pre-generation position, the chosen J-lens readout recovered nominated items strongly during permitted adoption and did not reliably recover them during abstention. Because unauthorised adoption never happened, the experiment does not show whether the readout separates taking part from refusing. Correct behaviour under the permitted prompt shows that the model can decode the tag in that condition; it does not show what computation ran under each refusal prompt, and failing to recover a token with one lens at one position is a limit of the measurement, not evidence that the information is absent. Two details of the implementation affect the numbers. The readout takes the best rank over every layer in a band and over four token variants of each item, so it is not a typical-layer rank. When all four candidates are censored the tie is broken in list order, so decoding rates in censored conditions include tie artefacts; the early-band rates of 0.30 to 0.37, with every median censored, are such artefacts and are left out of the table. This is an observation about one readout at one position. It says nothing about what the workspace contains.

## 6. Discussion

Each of the four findings points to an assumption an auditor has to check before trusting the evidence. Conditional dependence is evidence only relative to the conditioning information, and an innocent public rule the auditor has not modelled produces a confident false verdict (4.1). A corrected statistic is not a calibrated decision, and a rule calibrated on one family of nulls carried a 0.002 p-value onto a pair in a control with no planted coalition (4.2, 4.4). A pairwise correction built to remove third parties removes coalition members too, so for a coalition of three the published detector loses power as evidence accumulates (4.3). Ranking, thresholding and benefit are three different numbers, and a single accuracy figure hides that (4.4).

The new material is limited to the following. A general-N implementation of net influence with a within-state permutation null; the four evaluation axes studied together; the algebra of coalition cancellation; and two decision rules run side by side on two LLM pilot runs (scored with different procedures, as Section 4.4 says). Not new: the statistic itself, conditioning on state, permutation calibration of dependence statistics, scripted evaluation (Bonjour et al.; Lee et al.), and the observation that same-model agents correlate (Xu et al.). Section 5 is exploratory and shows neither that a proposal is present in the workspace nor that it is absent.

Future work, most useful first: a per-game permutation budget large enough for the Holm rule to be informative on single games (at least 120 permutations, preferably many more) and more games; a coalition detector built on a conditional-independence structure rather than pairwise net scores; a null that respects the agents' actual conditioning information, for example by fitting each agent's reaction function to the public history and permuting residuals; the Agrawal et al. (2025) double auction, where the public state is the order book; and, for the lens, a receiver that does adopt unauthorised proposals and a readout at the tag token rather than the decision token. Appendix A says what the results do and do not show.

## References

Agrawal, K. et al. (2025). Evaluating LLM Agent Collusion in Double Auctions. arXiv:2507.01413.

Ayeku, D. (2026). Telling Collusion from Common Knowledge: State-Conditioned Net Influence for LLM Agents. Apart Research AI Collusion Sprint entry (concurrent work).

Ayonrinde, K. and Lindsey, J. (2026). J++ Lens: Jacobian Filtering Enables More Faithful Workspace Lenses. LessWrong, 8 October 2026.

Berger, D. et al. (2025). Coordination by Code: Rethinking Antitrust Liability in the Age of AI Pricing. ABA Antitrust Magazine, Fall 2025.

Bonjour, T., Aggarwal, V. and Bhargava, B. (2022). Information theoretic approach to detect collusion in multi-agent games. UAI 2022, PMLR 180:223-232.

Ghanem, M. C. (2026). Steganalysis of Adaptive Covert Collusion in Tool-Using Agent Populations. arXiv:2608.02698.

Gurnee, W. et al. (2026). Verbalizable Representations Form a Global Workspace in Language Models. Transformer Circuits Thread, 6 July 2026.

Lee, C. E., Rosdalen, F., Samodelkin, P. and Edin, Y. (2026). From Pairs To Coalitions: Extending Information-Theoretic Collusion Detection To Multi-Agent Games. Apart Research AI Collusion Sprint entry (concurrent work).

Li, D., Varigonda, A., Feng, A., Chauhan, M. and Shrivastava, D. (2026). Collusion Does Not Localize to Jacobian-Lens Transport Subspaces. NeurIPS 2026, Third Workshop on Agents in the Wild: Safety, Security, and Beyond (poster; neurips.cc/virtual/2026/168189).

Rose, A., Cullen, C., Kaplowitz, B. G. and Schroeder de Witt, C. (2026). Detecting Multi-Agent Collusion Through Multi-Agent Interpretability. arXiv:2604.01151v1, 1 April 2026 (the privileged-observer control is in Section 4.4; later versions add authors and the HELIOS and NARCBench names).

Snow, J. (2026). Reading Moves, Not Messages: Detecting Collusion Between LLM Agents. Apart Research AI Collusion Sprint entry (concurrent work).

solarkyle (2026). jspace-lenses: fitted Jacobian lenses for open models. Hugging Face.

Tailor, O. (2025). Audit the Whisper: Detecting Steganographic Collusion in Multi-Agent LLMs. arXiv:2510.04303.

Veras Neves, C. E. (2026). j-space: Jacobian Lens for algorithmic collusion. Hugging Face Space (Trackio logbook), July 2026.

Xu, X., Wu, C., Lu, J., Tan, K., Tao, S. and Hong, H. (2026). Collusion with Competitive Marginals: Price-Level Audits Are Blind by Construction. arXiv:2607.26385.

## Appendix A. Limitations and Dual-Use / Ethical Considerations

**What the results show.** On scripted games with known ground truth: unconditioned dependence statistics rank an innocent same-rule pair first; majority-state conditioning removes that confound and fails against a rule keyed on a richer state; the permutation correction reduces but does not calibrate the fixed rule's false-positive rate at 25 rounds, and a Holm permutation rule is calibrated on the scripted nulls at a cost in power; and with two assistants the pairwise net correction cancels the coalition's signal, so the fixed rule's detection rate falls as the log grows. On the GPT-4.1-mini pilot (two runs of ten games per condition): the pair with a hidden channel is ranked first in eight or nine of ten games (upper bounds pending a tie audit) and flagged by the fixed rule in five or seven; its mean within-game score advantage was -0.25 and +1.00, which does not show a benefit; the fixed rule fires in half or more of single control games; and the Holm rule flags a pair on the pooled control without a planted coalition in both runs.

**What they do not show.** The scripted innocent rules are deterministic and key on states we can name; real innocent dependence (shared priors, common reactions to shocks nobody modelled) does not reduce to a small state. The collusion model assumes ex-ante signalling. The LLM pilot has one model, one temperature and ten games per condition per run. Run 1's pooled majority-state numbers were computed with cross-game state carry-over at the nine inter-game boundaries and its per-game rates do not measure any-pair false positives; run 2 corrects both, but run 1 could not be re-scored. The per-game Holm rule was not evaluated at adequate resolution (100 permutations). Ranking ties were not audited. The game bootstrap covers the uncorrected statistic only. "No planted coalition" is what the controls establish, not that no coordination of any kind took place. The permutation null treats same-state rounds as exchangeable, which is not established for agents conditioning on the full history, and the source of the dependence in the control (weights, history, position, strategy) is not identified. The lens assay has one receiver, one lens, a scripted sender, a four-item vocabulary and a planted codebook; its intended positive arm did not occur; its readout is a best-rank-over-band statistic with censoring; and it shows nothing about what the workspace contains. All code was written during the sprint and has not been reviewed by anyone else.

**Dual use.** A published account of where a detector fails tells a coalition where to hide: route coordination through a public signal the auditor has not modelled, keep per-round coordination below the floor measured here (CP about 0.2 at 25 rounds), or spread a scheme over three members so pairwise net scores cancel. We think publishing is justified because the same account tells an auditor which assumptions to check, and because every mechanism here is a property of the published statistic rather than a new attack. No real market data, deployed system or private information was used.

## Appendix B. Reproducibility and provenance

Code and logs: github.com/in-a-quantum-world/collusion-net-influence. All code was written during the sprint; nothing pre-existed. Completion times (UTC, equal to UK time on these dates): scripted main sweep 25 October 21:52; LLM pilot and lens notebooks started on Colab 25 October about 22:40; extensions 26 October about 00:00; lens run output recorded 26 October 09:10; LLM pilot run 1 output recorded 26 October 09:17; revised LLM analysis script 26 October about 09:50; LLM pilot run 2 (regenerated on Colab with identical code and settings) output recorded 26 October 11:06; this manuscript finalised 26 October 11:45, with a plain-language rewrite pushed to the repository afterwards and no change to any number. The submission deadline is 26 October 11:59 UTC. The final analysis commit is the last commit on main before 11:59 UTC. Run 2's raw game log (llm_games.jsonl) is to be added to results/20261026-1106-llm-pilot-run2/ from the Colab download as soon as possible after submission; its absence at submission is noted here. `python experiments/scripted_rps.py --seeds 100 --n_perm 100` regenerates the main sweep (about five minutes on two CPU cores); `python experiments/scripted_extensions.py --seeds 100 --n_perm 200` the extensions (about eleven minutes); `python experiments/llm_rps.py --model gpt-4.1-mini --games 10 --rounds 25` then `python experiments/analyse_llm.py --n_perm 500` the LLM pilot (under 2 USD); `collusion_llm_pilot.ipynb` and `lens_understanding_is_not_agreement.ipynb` wrap the LLM pilot and the lens assay for Colab, and `reanalyse_llm_pilot_v2.ipynb` re-scores an existing log. Tables: `results/scripted_tables.txt`, `results/extensions_tables.txt`; LLM and lens run outputs under `results/20261026-*`. Run 1 values in Table 3 are from its `llm_analysis.json` (original script); run 2 values in Tables 3 and 4 are from `llm_analysis_v2.json` (revised script), with the printed output under `results/20261026-1106-llm-pilot-run2/`.

## Appendix C. Lens assay details

Each base case fixes four items (red, blue, green, black), a policy marking two eligible and two ineligible, a random bijection from four tag words to the items (shown to the receiver in full), and a scripted sender nomination that is ineligible. Conditions: *unauthorized* (tag present, nomination ineligible, pair-bonus objective); *permitted* (same tag, nomination swapped into the eligible set); *refusal* (same tag, strict policy-compliance objective); *tag swapped* (tag for the other ineligible item, bonus objective); *tag erased* (tag line removed, bonus objective). Prompts are a single user turn ending "Reply with exactly one word: the item you approve (red, blue, green or black), or abstain"; generation is greedy, 6 tokens. The lens is applied at the last prompt token. For each candidate item we take the minimum, over layers in the band and over four token variants (bare, leading space, capitalised, both), of the position of the item's first token in the lens logits sorted descending, truncated at 5,000 (censored beyond). Bands: early = layers 0 to 10, late = layers 28 to 43, upper = layers 11 to 46, of 47. Candidate decoding is whether the transmitted item has the smallest rank of the four, with list-order tie-breaking; this inflates decoding of "red" when all four are censored, which is why the early band is not reported. Outputs: unauthorized, refusal, tag swapped and tag erased produced "abstain" in 30 of 30 cases each; permitted produced the nominated item in 30 of 30; no output was malformed. Upper-band decoding of the transmitted item was 0.10, 1.00, 0.07, 0.03 and 0.00 for the five conditions in table order, with median nomination ranks 23, 0, 289, 91 and 217.

![Figure 3. The lens assay: scripted sender, random codebook, receiver prompt, lens readout at the last prompt token, one-word action, and the five matched conditions per base case.](../results/fig_lens_assay.png){width=100%}

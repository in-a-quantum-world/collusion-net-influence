---
title: "When Net Influence Misleads: Calibration, Innocent Dependence, and Coalition Size"
author: "Rucha Agashe, Imperial College London (ragashe1618@gmail.com)"
date: "AI Collusion Research Sprint, Apart Research, 23 to 25 October 2026 (submissions close 25 October 11:59 PM AoE, which is 26 October 11:59 UTC). Track 1 (question 1.4) and Track 2 (question 2.3)."
---

## Abstract

Pairwise net influence (Bonjour et al. 2022) detects a colluding pair from action logs alone: dependence between two agents' actions conditional on the public state, minus the strongest third-party dependence. We ask under which assumptions about public information, sample size and coalition structure it distinguishes planted coordination from innocent dependence, and where calibration fails to transfer. We extend the original ground-truth evaluation to matched innocent-dependence confounds, short-horizon calibration, state misspecification and a three-member coalition, over 7,400 scripted games, and then test transfer on 30 games of GPT-4.1-mini agents with and without a hidden private channel. Four failure mechanisms emerge. First, conditioning on the wrong public state confidently identifies an innocent pair that follows a shared public rule, in every game. Second, reducing the plug-in bias of the statistic is not the same as controlling false positives: the fixed threshold fires in half of all-random 25-round games, and a Holm-corrected permutation rule that is calibrated on the scripted nulls flags an innocent same-model pair on the pooled LLM control. Third, with a primary and two assistants, the pairwise net correction subtracts one coalition member's influence from another's and cancels the signal in the population limit, so more data makes the fixed-threshold detector less likely to fire. Fourth, on the LLM pilot, ranking, detection and benefit come apart: the team is ranked first in nine of ten games, flagged in five, and gained nothing. An exploratory Jacobian-lens assay on gemma-4-12B-it is reported with its intended positive arm absent. Code and logs are released.

## 1. Introduction

The sprint asks what should count as evidence that agents are colluding when the evidence is a log of actions (question 2.3), and whether information-theoretic measures can identify colluding pairs among several agents (question 1.4). Antitrust law has long faced the same difficulty: competitors act in parallel because they respond to the same public information, and parallel conduct alone is not an agreement (Berger et al. 2025). A detector that flags correlation flags every pair that shares a prior or a strategy, and two instances of one base model share a great deal of prior (Xu et al. 2026).

Bonjour et al. (2022) proposed a statistic built for exactly this difficulty. It measures the dependence between two agents' actions conditional on the observable game state, then subtracts the strongest dependence any third agent has on the same target. It is domain independent and needs only logged actions. The original paper evaluated it on known colluding configurations, varied collusion strength and sample size, treated the threshold as a hyperparameter to be tuned on training data, and limited its scope to two colluders. Several concurrent sprint entries apply it to LLM players (Ayeku 2026; Snow 2026; Lee et al. 2026), and Lee et al. include 2,800 scripted coalition games. None of this tells an auditor which assumptions the verdict rests on. That is the question here:

Under which assumptions about public information, sample size and coalition structure does net influence distinguish planted coordination from innocent dependence, and where does calibration fail to transfer?

We extend prior ground-truth evaluation along four axes that the original paper and the concurrent entries do not cover: a matched innocent-dependence confound (two non-colluders following the same public rule), short-horizon calibration (25 to 500 rounds, where plug-in bias is comparable to the published threshold), state misspecification (an innocent rule keyed on a richer state than the detector conditions on) and a three-member coalition. We then apply both the fixed-threshold rule and a permutation-calibrated rule to GPT-4.1-mini agents with a hidden private channel and to a same-model control, and report ranking, detection and coalition benefit as separate outcomes. The principle throughout is the one the sprint brief states: do not confuse an observable signal with evidence of collusion. Section 5 reports, as a bounded exploratory result, a Jacobian-lens assay that asked the same question of a model's internals and whose intended positive arm did not occur.

## 2. Related work

**Information-theoretic detection from actions.** Bonjour, Aggarwal and Bhargava (2022) define influence between agents as conditional mutual information given the game state and flag a pair whose net influence exceeds a threshold tuned on labelled data; they evaluate on iterated Rock-Paper-Scissors and poker with two colluders, across collusion probabilities and numbers of games, and note the two-colluder scope as a limitation. Later information-theoretic detectors for LLM agents (Tailor 2025; Ghanem 2026) estimate cross-run mutual information over transcripts rather than actions. Xu et al. (2026) show that two deployments of one base model carry a small residual bid correlation and argue that price-level audits are blind by construction. Within this sprint, Ayeku (2026) and Snow (2026) apply net influence with a permutation test to LLM players, Ayeku conditions on a public state against an innocent rule-following pair, and Lee et al. (2026) extend the statistic to coalitions with graph methods over 2,800 scripted coalition games and 400 honest games. Our axes differ: matched innocent-dependence confounds under correct and misspecified conditioning, the gap between bias correction and decision calibration at short horizons, an analytic account of why the pairwise correction cancels a three-member coalition, and a transfer test in which both decision rules are run on the same LLM logs.

**Detection from internals.** Rose et al. (2026) train a deception direction on residual activations and aggregate per-agent scores across a group; in their limitations (v1, Section 4.4) they write that a privileged-observer control, an agent who knows the collusion scheme but is instructed not to participate, would help, and leave it for future work. Li et al. (2026) find that a probe confined to the Jacobian-lens transport subspace detects an agent's stated stance but not withheld collusion. Veras Neves (2026) reads a coordination signal from the J-lens on single prompts. Gurnee et al. (2026) and Ayonrinde and Lindsey (2026) report that a concept can remain in the workspace while the model is instructed to suppress it. Section 5 is a small exploratory step toward the control Rose et al. describe.

## 3. Method

### 3.1 Pairwise net influence

Consider N agents playing a repeated simultaneous-move game with K actions. Let A_i^t be agent i's action in round t and S^t the public state before round t. Bonjour et al. define the influence of i on j as

gamma(i; j) = sum_s p(s) sum_{a_i, a_j} p(a_i, a_j | s) log [ p(a_i, a_j | s) / ( p(a_i | s) p(a_j | s) ) ],

the conditional mutual information I(A_i; A_j | S), symmetric in the simultaneous fully observable case. The net influence is

Gamma(i; j) = gamma(i; j) - max_{k not in {i, j}} gamma(k; j),

and agents i and j are declared colluding if Gamma(i; j) >= alpha and Gamma(j; i) >= alpha. The paper reports a pair only when exactly one pair satisfies this and uses alpha = 0.05 (natural log). We implement this with plug-in probabilities and score each unordered pair by min(Gamma(i; j), Gamma(j; i)).

### 3.2 State functions

We compare a memoryless state (gamma reduces to mutual information), the previous round's majority action (four values) and the full previous action profile (82 values, sparse at short horizons).

### 3.3 Bias correction and two decision rules

Plug-in mutual information is biased upwards by about (K-1)^2 / (2T) nats per state stratum; for K = 3 and T = 25 this is 0.08 nats, larger than alpha. We estimate the null of gamma(i; j) under "A_j independent of A_i given S" by shuffling A_j within each state stratum (100 to 500 permutations) and report the corrected statistic gamma minus the null mean, clipped at zero, and a permutation p-value per pair. Net influence is computed from the corrected gamma. Two decision rules are then applied, and they are different procedures: the **fixed rule** (exactly one pair has net-influence score >= 0.05) and the **Holm rule** (the top-scoring pair has a Holm-adjusted permutation p-value below 0.05 over the six pairs). Rejecting conditional independence is a statement about a specified null given the conditioning information; it is not equivalent to establishing collusion, and Section 4 is about the gap between the two.

### 3.4 Scripted environment

Four players (five in Section 4.3), actions rock, paper, scissors. Payoffs follow Bonjour et al.: all the same, nobody scores; all three actions present, everyone scores 1; exactly two actions present, the winning action's players score 1. Agent types: *random*; *assistant*, which with probability CP plays the action that loses to the primary's action (guaranteeing the primary a point, with ex-ante signalling assumed as in the paper) and is otherwise uniform; *parallel*, which plays the action that beats the previous round's majority (a public, deterministic, unilateral rule); *parallel_rich*, which plays the action that beats the previous move of the lowest-indexed player who scored in the previous round (a function of the previous full profile but not of the previous majority). Conditions: *bonjour* (A random, B assistant, C and D random), *parallel* (C and D parallel; innocent pair CD), *null* (all random), *parallel_only* (no collusion, C and D parallel), *parallel_rich* and *parallel_rich_only*, and a five-player *coalition* condition (A primary, B and C assistants, D and E random). Main sweep: T in {25, 50, 100, 200, 500} at CP = 0.6 and CP in {0, 0.2, 0.4, 0.6, 0.8, 1.0} at T = 25 and 100, 100 seeds per cell, 100 permutations (4,400 games, seeds 0 to 99 and 10000 to 10099). Extensions: 100 seeds per cell, 200 permutations, different seeds (3,000 games). Nominally identical cells therefore differ slightly between the two batches (for example a parallel-only false-positive rate of 0.25 in the main sweep against 0.33 in the extensions at T = 25); both are reported with their batch.

Detectors, all action-only: net influence with each state, with and without correction; raw mutual information; Cramer's V; same-action rate. Outcomes, kept separate: rank-1 accuracy (the true pair scores highest of six), correct detection (a rule flags exactly the true pair), and false-positive rate (a rule flags any pair when no collusion is present).

## 4. Results

![Figure 1. Main sweep (100 seeds per point). (a) Rank-1 accuracy of the true pair when an innocent parallel pair is present, CP = 0.6. (b) False-positive rate of the fixed rule with no collusion present, in the all-random and parallel-only worlds. (c) Correct detection at 25 rounds against collusion strength, random non-colluders.](../results/fig_scripted.png){width=100%}

**Reproduction.** With random non-colluders, the paper's setting, memoryless net influence behaves as published: rank-1 accuracy at CP = 0.6 is 0.95 at 25 rounds and 1.00 from 50; correct detection under the fixed rule is 0.82, 0.93 and 1.00 at 25, 50 and 100 rounds, and at T = 100 rises from 0.24 at CP = 0.2 to 1.00 at CP = 0.6. Same-action rate scores 0.00 in every collusion condition from 50 rounds, because the assisting move is the move that loses to the partner; Cramer's V and raw mutual information, which are sign-agnostic, match net influence here.

### 4.1 State misspecification: the wrong public state confidently identifies an innocent pair

In the *parallel* condition, memoryless net influence, raw mutual information, Cramer's V and same-action rate all rank the innocent pair CD first in 96 to 100% of games at every T and every CP below 1.0 (Table 1); gamma(C; D) is about 1.07 nats against 0.17 for the true pair at CP = 0.6, and the net correction does not help because no third party influences A or B. The fixed rule names the innocent pair in 83 to 100% of *parallel_only* games. Conditioning on the previous majority removes the dependence exactly (gamma(C; D | S) = 0), rank-1 accuracy of the true pair becomes 1.00 at every T and CP from 0.2, and the innocent pair is never flagged.

That success depends on conditioning on the state the innocent rule uses. Against *parallel_rich*, majority-state conditioning fails completely: gamma(C; D | S) stays at 0.91 to 1.07 nats, the innocent pair ranks first in 100% of games at every T, and the fixed rule flags it in 65 to 100% of no-collusion games. Conditioning on the full previous profile removes the dependence but spreads 25 to 500 rounds over 15 to 29 occupied states; uncorrected, the bias flags a pair in 97 to 100% of no-collusion games, and even corrected the true pair is ranked first in only 0.75 of 25-round games (0.97 at 50, 1.00 from 100; false-positive rate 0.00 to 0.06). An auditor who conditions on a coarser state than the agents' actual public rule obtains a confident, wrong verdict, and one who conditions on a richer state pays in sample size.

**Table 1. Rank-1 accuracy of the true pair, CP = 0.6, innocent parallel pair present (main sweep).**

| T | NI memoryless | NI majority | NI+perm memoryless | NI+perm majority | raw MI | same-action | Cramer's V |
|---|---|---|---|---|---|---|---|
| 25 | 0.00 | 1.00 | 0.00 | 0.98 | 0.00 | 0.00 | 0.00 |
| 50 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| 100 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| 200 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| 500 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 |

### 4.2 Bias correction is not decision calibration

Once C and D are deterministic given S, gamma(C; j | S) is zero for every j, the net correction subtracts nothing from the random pair AB, and Gamma(A; B) is pure estimation bias. In *parallel_only* the state-conditioned fixed rule flags a pair in 100% of 25-round games and 67% of 100-round games; at CP = 0 in *parallel* it names the innocent random pair AB in 100% of 25-round games. Subtracting the within-state permutation null mean reduces the *parallel_only* rate to 0.25, 0.20, 0.05 and 0.01 at 25, 50, 100 and 200 rounds while rank-1 accuracy stays at 0.98 to 1.00. But in the all-random *null* world the corrected fixed rule still fires in 0.49 to 0.55 of 25-round games and 0.01 to 0.10 of 100-round games: removing the mean of the bias leaves its variance, and the fixed threshold has no account of it.

The Holm rule is calibrated on the scripted nulls where the fixed rule is not (Table 2): 0.03 to 0.06 in the all-random world at every length and 0.00 to 0.01 in the parallel-only world, never naming the innocent pair. The price is power at short horizons: 0.42 against 0.96 in *parallel* and 0.38 against 0.76 in *bonjour* at 25 rounds; from 100 rounds both rules reach 1.00. Section 4.4 shows that this calibration does not transfer to the LLM control.

**Table 2. Fixed rule versus Holm rule, NI+perm, state = previous majority, CP = 0.6 (extensions batch).**

| T | fixed: power (parallel) | fixed: FPR (random) | fixed: FPR (parallel-only) | Holm: power (parallel) | Holm: FPR (random) | Holm: FPR (parallel-only) |
|---|---|---|---|---|---|---|
| 25 | 0.96 | 0.49 | 0.33 | 0.42 | 0.03 | 0.00 |
| 50 | 1.00 | 0.24 | 0.21 | 0.87 | 0.03 | 0.00 |
| 100 | 1.00 | 0.04 | 0.06 | 1.00 | 0.04 | 0.00 |
| 200 | 1.00 | 0.01 | 0.00 | 1.00 | 0.06 | 0.01 |

### 4.3 Coalition cancellation: the pairwise correction removes a three-member coalition's signal

Bonjour et al. limit their method to two colluders. In the five-player condition with a primary A and two assistants B and C (CP = 0.6), the mechanism that makes net influence robust to third parties works against the coalition. In the symmetric construction gamma(B; A) = gamma(C; A) = m in expectation, so

Gamma(B; A) = gamma(B; A) - max_k gamma(k; A) = m - m = 0

whenever neither outsider influences A more than the other assistant does, and since the pair score is the minimum of the two directions, the true pair's score cannot exceed a positive threshold in the population limit. The subtraction treats another coalition member as the competing explanation and removes the signal. Empirically the AB score falls to 0.006 nats at 500 rounds and the fixed rule returns no pair in 25%, 32%, 54% and 68% of games at 50, 100, 200 and 500 rounds, almost always because no pair crosses alpha; detection gets worse as the log grows. Ranking is unaffected: a true pair is ranked first in 0.98 to 1.00 of games under both raw mutual information and net influence, and raw mutual information places all three coalition pairs in the top three from 200 rounds (2.81 of 3 at 50). Net influence places exactly two, because B and C are conditionally independent given A (gamma(B; C | A) = 0.014 at 500 rounds) and Gamma(B; C) is negative (-0.24 nats). Whether that is right depends on the target: B and C are members of the same coalition while having no direct dependence, and coalition membership and direct interaction edges are different ground truths. A coalition detector needs a conditional-dependence structure over the group, not a pairwise net score.

![Figure 2. Extensions (100 seeds per point). (a) Fixed rule versus Holm rule. (b) Innocent pair keyed on the full previous profile: majority-state conditioning fails; profile-state conditioning with correction recovers the true pair from 50 rounds. (c) Coalition of three: fraction of the three true pairs in the top three under raw mutual information and net influence.](../results/fig_extensions.png){width=100%}

### 4.4 Transfer: the LLM pilot

**Setup.** Four GPT-4.1-mini players (temperature 1.0), 25 rounds, full public history in context; ten games in each of three conditions: *collude* (Alpha and Bravo told privately they are a team maximising their combined score, with one private message each before every round), *team_no_channel* (same instruction, no channel) and *independent* (no team; a same-model control). The detector sees only the 25 x 4 action log; messages are logged but never scored. Outcomes are kept separate: pair ranking, detection under each rule, and coalition benefit (team mean score minus non-team mean score).

**Table 3. LLM pilot, pooled 250 rounds per condition, memoryless state, permutation-corrected scores (values from the run's llm_analysis.json; Alpha-Bravo is the team).**

| condition | coalition benefit per game | gamma(Alpha; Bravo), nats (perm p) | team rank of 6 | fixed rule flags | top-scoring pair (perm p, Holm p) | Holm rule flags |
|---|---|---|---|---|---|---|
| collude | -0.25 | 0.166 (0.002) | 1 | Alpha-Bravo | Alpha-Bravo (0.002, 0.012) | Alpha-Bravo |
| team_no_channel | -0.20 | 0.021 (0.046) | 2 | none | Charlie-Delta (0.002, 0.012) | Charlie-Delta, a non-team pair |
| independent | +0.75 (no team) | 0.020 (0.052) | 4 | none | Alpha-Delta (0.002, 0.012) | Alpha-Delta, an innocent pair |

**Table 4. LLM pilot, per game (10 games of 25 rounds per condition; from the run's llm_analysis.json).**

| condition | memoryless: team ranked first | memoryless: fixed rule flags the team | majority state, corrected: team ranked first | majority state, corrected: fixed rule flags the team |
|---|---|---|---|---|
| collude | 9/10 | 5/10 | 3/10 | 0/10 |
| team_no_channel | 1/10 | 0/10 | 1/10 | 0/10 |
| independent | 0/10 | 0/10 | 1/10 | 0/10 |

**Ranking, detection and benefit come apart.** With the private channel, the team pair has the largest pairwise dependence by a factor of four and is ranked first on the pooled log and in nine of ten single games under the memoryless statistic. The fixed rule flags it on the pooled log but in only five of ten single games; with majority-state conditioning the team is ranked first in three games and flagged in none, the power cost of splitting 25 rounds over four states that Section 4.2 predicted. The private messages show what the statistic responds to, a sacrificial scheme in which one partner plays the move that loses to the other ("I'll play rock to beat your scissors, so I can score while you lose"), alternating with same-move rounds. The team gained nothing from it: coalition benefit was -0.25 points per game. A single 25-round game therefore gives a confident ranking, an unreliable detection and no demonstrated harm, and "90% detection" would describe none of these correctly. Without the channel the team's dependence is 0.021 nats, ranked second, and is not flagged by either rule.

**Calibration does not transfer.** Applying the Section 4.2 Holm rule to the pooled logs, the top-scoring pair has a raw permutation p-value of 0.002 (Holm 0.012) in every condition: the team in *collude*, Charlie-Delta in *team_no_channel*, Alpha-Delta in *independent*. The rule that was calibrated on the scripted nulls flags a non-team pair in the no-channel condition and an innocent pair in the control with no team at all. The fixed rule flags nothing in those two conditions, but narrowly: Alpha-Delta's net score in the control is 0.037 against the threshold of 0.05. Permutation calibration controlled rejection rates in the tested scripted nulls; it did not distinguish innocent dependence from planted coordination in the pooled LLM pilot. The dependence between those innocent pairs is real under the permutation null, but our design does not say what causes it: shared model weights (Xu et al. 2026), common public history, position effects or shared strategies would all produce it, and the conditioning on the previous majority alone does not make same-state rounds exchangeable for agents that condition on the full history.

**What the per-game numbers do and do not say.** The per-game hit rates in Table 4 measure whether the designated team was flagged, not whether any pair was; a zero in the independent condition does not establish a zero per-game false-positive rate. The run's pooled majority-state numbers were also computed with the state sequence recomputed across game boundaries, so the first round of games two to ten received a state carried over from a different game (10 of 250 rounds). A revised analysis script in the repository (experiments/analyse_llm.py) computes states per game before pooling, restricts permutations to within-game strata, reports any-pair false positives, correct-pair flags and ranking separately under both rules, and attaches game-level uncertainty (Wilson intervals on per-game rates, bootstrap over games on pooled scores); it was written after the run and before submission and is to be applied to the released logs.

## 5. Exploratory: a Jacobian-lens readout of a coded proposal

As a bounded secondary experiment we asked whether a receiver model's workspace representation of a partner's coded proposal differs between adopting it and understanding it and declining, the privileged-observer contrast Rose et al. (2026) describe. Design, bands and metrics were specified in code before these runs; full details are in Appendix C. Receiver gemma-4-12B-it (4-bit, solarkyle Jacobian lens, read at the last prompt token) saw a policy marking two of four items ineligible, a random four-word codebook, a routing tag from a scripted partner naming an ineligible item, and an objective; it replied with one item or abstain. Five matched prompts per base case (unauthorized, permitted, refusal, tag swapped, tag erased), 30 base cases.

**Table 5. Lens assay (30 base cases per condition). Decoding is "transmitted item ranked first of the four candidates" using the best rank across layers in the late band (layers 28 to 43 of 47) and across token variants; ranks beyond 5,000 are censored.**

| condition | adopted nomination | policy violation | abstain | late band: transmitted item decoded | late band: median rank of nomination |
|---|---|---|---|---|---|
| unauthorized | 0/30 | 0/30 | 30/30 | 0.23 | 1383 |
| permitted | 30/30 | 0/30 | 0/30 | 1.00 | 0 |
| refusal | 0/30 | 0/30 | 30/30 | 0.17 | censored |
| tag swapped | 0/30 | 0/30 | 30/30 | 0.13 | 1987 |
| tag erased | 0/30 | 0/30 | 30/30 | n/a | censored |

At the final pre-generation position, the selected J-lens readout strongly recovered nominated items during permitted adoption but did not reliably recover them during abstention. Because unauthorized adoption never occurred, this experiment does not establish whether the readout distinguishes collusive participation from refusal. Correct behaviour under the permitted prompt shows decoding capability in that condition; it does not verify the computation under each refusal prompt, and failure to recover a token with one lens at one position is a measurement limitation, not evidence that the information is absent. Two implementation details bear on the numbers: the readout takes the best rank across all layers in a band and across four token variants of each item, not a typical-layer rank; and when all four candidates are censored the tie is broken in list order, so decoding rates in censored conditions include tie artefacts (the early-band rates of 0.30 to 0.37, with every median censored, are such artefacts and are omitted from the table). We report this as an observation about one readout at one position and claim nothing about what the workspace contains.

## 6. Discussion

The four mechanisms in Section 4 share one lesson: each plausible piece of collusion evidence rests on an assumption an auditor must check before trusting it. Conditional dependence is evidence only relative to the conditioning information, and an innocent public rule the auditor has not modelled produces a confident false verdict (4.1). A corrected statistic is not a calibrated decision, and a decision rule calibrated on one family of nulls carried a 0.002 p-value onto an innocent same-model pair in another (4.2, 4.4). A pairwise correction designed to remove third parties removes coalition members too, so for a coalition of three the published detector loses power as evidence accumulates (4.3). And ranking, thresholding and benefit are different outcomes that a single number conflates (4.4).

What is new here is narrow. We release a general-N implementation of net influence with a within-state permutation null; we extend the original ground-truth evaluation to matched innocent-dependence confounds, short-horizon calibration, state misspecification and a three-member coalition; we give the algebraic account of coalition cancellation; and we run two decision rules side by side on the same LLM logs. Not new: the statistic, conditioning on state, permutation calibration of dependence statistics, scripted evaluation as such (Bonjour et al.; Lee et al.), and the observation that same-model agents correlate (Xu et al.). Section 5 is exploratory and establishes neither presence nor absence of a proposal in the workspace.

Future work, in order of value: apply the revised per-game analysis to the released logs and report any-pair false positives; a coalition detector built on a conditional-independence structure rather than pairwise net scores; a null that respects the agents' actual conditioning information, for example by fitting each agent's reaction function to the public history and permuting residuals; the Agrawal et al. (2025) double auction, where the public state is the order book; and, for the lens, a receiver that does adopt unauthorised proposals and a readout at the tag token rather than the decision token.

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

Rose, A., Cullen, C., Abdelnabi, S., Torr, P., Kaplowitz, B. G. and Schroeder de Witt, C. (2026). Detecting Multi-Agent Collusion Through Multi-Agent Interpretability. arXiv:2604.01151 (v1, 1 April 2026; the privileged-observer control is in Section 4.4).

Snow, J. (2026). Reading Moves, Not Messages: Detecting Collusion Between LLM Agents. Apart Research AI Collusion Sprint entry (concurrent work).

solarkyle (2026). jspace-lenses: fitted Jacobian lenses for open models. Hugging Face.

Tailor, O. (2025). Audit the Whisper: Detecting Steganographic Collusion in Multi-Agent LLMs. arXiv:2510.04303.

Veras Neves, C. E. (2026). j-space: Jacobian Lens for algorithmic collusion. Hugging Face Space (Trackio logbook), July 2026.

Xu, X., Wu, C., Lu, J., Tan, K., Tao, S. and Hong, H. (2026). Collusion with Competitive Marginals: Price-Level Audits Are Blind by Construction. arXiv:2607.26385.

## Appendix A. Limitations and Dual-Use / Ethical Considerations

**What the results establish.** On scripted games with known ground truth: unconditioned dependence statistics rank an innocent same-rule pair first; majority-state conditioning removes that confound and fails against a rule keyed on a richer state; the permutation correction reduces but does not calibrate the fixed rule's false-positive rate at 25 rounds, and a Holm permutation rule is calibrated on the scripted nulls at a power cost; and with two assistants the pairwise net correction cancels the coalition's signal, so the fixed rule's detection rate falls with sample size. On the GPT-4.1-mini pilot (ten games per condition): the team with a hidden channel is ranked first in nine of ten games and flagged by the fixed rule in five; it gained no benefit; and the Holm rule flags a non-team pair in the no-channel condition and an innocent pair in the control.

**What they do not establish.** The scripted innocent rules are deterministic and key on states we can name; real innocent dependence (shared priors, common reactions to unmodelled shocks) is not reducible to a small state. The collusion model assumes ex-ante signalling. The LLM pilot has one model, one temperature and ten games per condition; its pooled majority-state numbers were computed with cross-game state carry-over at 10 of 250 rounds; its per-game hit rates do not measure any-pair false positives; and the permutation null treats same-state rounds as exchangeable, which is not established for agents conditioning on the full history. The source of the innocent dependence in the control (weights, history, position, strategy) is not identified. The lens assay has one receiver, one lens, a scripted sender, a four-item vocabulary and a planted codebook, its intended positive arm did not occur, its readout is a best-rank-over-band statistic with censoring, and it establishes nothing about what the workspace contains. All code was written during the sprint and has not been independently reviewed.

**Dual use.** A published account of where a detector fails tells a coalition where to hide: route coordination through a public signal the auditor has not modelled, keep per-round coordination below the detection floor measured here (CP about 0.2 at 25 rounds), or spread a scheme over three members so pairwise net scores cancel. We think publication is justified because the same account tells an auditor which assumptions to check, and because every mechanism here is a property of the published statistic rather than a new attack. No real market data, deployed system or private information was used.

## Appendix B. Reproducibility and provenance

Code and logs: github.com/in-a-quantum-world/collusion-net-influence. All code was written during the sprint; nothing pre-existed. Completion times (UTC, which equals UK time on these dates): scripted main sweep 25 October 21:52; LLM pilot and lens notebooks started on Colab 25 October about 22:40; extensions 26 October about 00:00; lens run output recorded 26 October 09:10; LLM pilot output recorded 26 October 09:17; revised LLM analysis script and this manuscript 26 October before 11:00. The submission deadline is 26 October 11:59 UTC. `python experiments/scripted_rps.py --seeds 100 --n_perm 100` regenerates the main sweep (about five minutes on two CPU cores); `python experiments/scripted_extensions.py --seeds 100 --n_perm 200` the extensions (about eleven minutes); `python experiments/llm_rps.py --model gpt-4.1-mini --games 10 --rounds 25` then `python experiments/analyse_llm.py --n_perm 500` the LLM pilot (under 2 USD); `collusion_llm_pilot.ipynb` and `lens_understanding_is_not_agreement.ipynb` wrap the LLM pilot and the lens assay for Colab. Tables: `results/scripted_tables.txt`, `results/extensions_tables.txt`; LLM and lens run outputs under `results/20261026-*`. Tables 3 and 4 are taken from the run's `llm_analysis.json` produced by the original analysis script; the revised script writes `llm_analysis_v2.json`.

## Appendix C. Lens assay details

Each base case fixes four items (red, blue, green, black), a policy marking two eligible and two ineligible, a random bijection from four tag words to the items (shown to the receiver in full), and a scripted sender nomination that is ineligible. Conditions: *unauthorized* (tag present, nomination ineligible, pair-bonus objective); *permitted* (same tag, nomination swapped into the eligible set); *refusal* (same tag, strict policy-compliance objective); *tag swapped* (tag for the other ineligible item, bonus objective); *tag erased* (tag line removed, bonus objective). Prompts are a single user turn ending "Reply with exactly one word: the item you approve (red, blue, green or black), or abstain"; generation is greedy, 6 tokens. The lens is applied at the last prompt token; for each candidate item we take the minimum, over layers in the band and over four token variants (bare, leading space, capitalised, both), of the position of the item's first token in the lens logits sorted descending, truncated at 5,000 (censored beyond). Bands: early = layers 0 to 10, late = layers 28 to 43, upper = layers 11 to 46, of 47. Candidate decoding is whether the transmitted item has the smallest rank of the four, with list-order tie-breaking; this inflates decoding of "red" when all four are censored, which is why the early band is not reported. Outputs: unauthorized, refusal, tag swapped and tag erased produced "abstain" in 30 of 30 cases each; permitted produced the nominated item in 30 of 30; no output was malformed. Upper-band decoding of the transmitted item was 0.10, 1.00, 0.07, 0.03 and 0.00 for the five conditions in table order, with median nomination ranks 23, 0, 289, 91 and 217.

![Figure 3. The lens assay: scripted sender, random codebook, receiver prompt, lens readout at the last prompt token, one-word action, and the five matched conditions per base case.](../results/fig_lens_assay.png){width=100%}

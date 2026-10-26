---
title: "Correlation is not agreement, and the workspace shows the action: calibrated net influence and a Jacobian-lens control for AI agent collusion"
author: "Rucha Agashe, Imperial College London (ragashe1618@gmail.com)"
date: "AI Collusion Research Sprint, Apart Research, 25 October 2026. Track 1 (question 1.4); also relevant to Track 2 (question 2.3)."
---

## Abstract

What should count as evidence that two AI agents are colluding? Antitrust law holds that parallel conduct is not an agreement and demands plus factors. We test one plus factor at two evidence surfaces, action logs and model internals, and in each case measure how the naive reading mistakes something innocent for collusion. On actions, we implement pairwise net influence (Bonjour et al. 2022) for N players, add a within-state permutation null, and run the ground-truth evaluation the method has never had: 7,400 scripted games over five horizons, six collusion strengths, four baselines, two public-state definitions, two kinds of innocent pair and a coalition of three. The published memoryless statistic and every unconditioned baseline rank the innocent pair first in 96 to 100 percent of games; conditioning on the public state recovers the true pair in 98 to 100 percent; the permutation null cuts the 25-round false-positive rate from 100 percent to 25 percent; and we report where the conditioned detector breaks. On internals, we run the privileged-observer control Rose et al. (2026) proposed: a Jacobian-lens assay on gemma-4-12B-it with a partner's coded proposal and five matched conditions. The receiver never approved an ineligible item; the proposal reads out at vocabulary rank 0 when the receiver is about to act on it legitimately and at chance when it has understood it and declined. A workspace monitor at the decision point sees the action, not the understanding. Code, logs and notebooks are released.

## 1. Introduction

The sprint asks what should count as evidence that agents are colluding when the evidence is a log of actions (question 2.3), and whether information-theoretic measures can identify colluding pairs among several agents (question 1.4). Both questions share a difficulty that antitrust law has faced for decades: competitors frequently act in parallel because they respond to the same public information, and parallel conduct alone is not an agreement. A detector that flags correlation flags every pair that shares a prior or a strategy. This matters for LLM agents in particular, because two instances of the same base model share a great deal of prior (Xu et al. 2026), so the base rate of innocent correlation is high.

Bonjour et al. (2022) proposed a statistic that is attractive for exactly this reason. Rather than raw dependence, it measures the dependence between two agents' actions conditional on the observable game state, and then subtracts the strongest dependence any third agent has on the same target. It is domain independent and needs only logged actions. Before this sprint it had been tested only on three scripted or Q-learning players, with no baseline comparison, no public code, and a fixed threshold whose behaviour at short horizons was not examined. Several concurrent sprint entries apply it directly to LLM players (Ayeku 2026; Snow 2026; Lee et al. 2026). What none of them provides, and what a detector needs before anyone relies on its verdicts, is a ground-truth characterisation: at which horizons, collusion strengths and kinds of innocent correlation does it hold, and where does it break. That is the first half of this report.

The same problem recurs one level down. Interpretability-based collusion detectors (Rose et al. 2026, with the HELIOS framework and NARCBench) read a coalition out of activations. But a receiver that understands a partner's harmful proposal in order to refuse it must represent that proposal too, so a lens that finds the proposal in the workspace has not yet found an agreement. Rose et al. name exactly this control, a privileged observer that knows the scheme but does not take part, and leave it to future work; Gurnee et al. (2026) and Ayonrinde and Lindsey (2026) show that a concept can sit in the workspace while the model suppresses it. The second half of this report runs that control with a workspace lens.

This report therefore addresses the plus-factor logic at two evidence surfaces, in a weekend-sized scope:

1. Actions. We implement pairwise net influence for arbitrary N with a permutation null, and evaluate it with ground truth against four baselines across horizons, collusion strengths, two public-state definitions, an innocent same-rule pair, a harder innocent pair keyed on a richer state, and a coalition of three. A harness to apply the same detector to GPT-4.1-mini agents with a hidden private channel is built and released, but its results did not arrive before the deadline.
2. Internals. Using a fitted Jacobian lens on gemma-4-12B-it, we run the privileged-observer control: does the workspace readout of a partner's coded proposal differ between a receiver that adopts it and one that understands it and refuses, with matched permission, tag-swap and tag-erase conditions.

The scripted environment costs nothing and gives ground truth, which a detector needs before anyone trusts it on LLM logs; the LLM and lens harnesses are built to test the two surfaces on real models and will be run after the sprint. Figure 1 summarises the design.

![Figure 1. Two evidence surfaces, one rule. Left: on an action log, the naive statistic points at the innocent pair that follows the same public rule; the state-conditioned, permutation-calibrated statistic points at the hidden coalition. Right: in the receiver's workspace, the lens reads the partner's proposal; the assay asks whether the readout differs between adopting and understanding-and-refusing.](../results/fig_evidence_ladder.png){width=80%}

## 2. Related work

**Information-theoretic detection from actions.** Bonjour, Aggarwal and Bhargava (2022) define influence between agents as conditional mutual information given the game state and flag a pair whose net influence exceeds a fixed threshold; the method was validated on three scripted or Q-learning players with no baseline and no public code. Later information-theoretic detectors for LLM agents (Tailor 2025; Ghanem 2026) estimate cross-run mutual information over transcripts, not actions, and do not condition on a public state. Xu et al. (2026) show that two deployments of one base model carry a small residual bid correlation, the LLM form of the parallel-conduct confound we test. Within this sprint, Ayeku (2026) and Snow (2026) apply net influence with a permutation test to LLM players, and Ayeku conditions on a public state against an innocent rule-following pair; Lee et al. (2026) extend it to coalitions with graph methods. Our scripted sweep is the systematic ground-truth evaluation those LLM-first entries do not report.

**Detection from internals.** Rose et al. (2026) train a deception direction on residual activations and aggregate per-agent scores across a group, reaching 1.00 AUROC in-distribution and 0.73 to 0.93 zero-shot. They name, and leave to future work, a privileged-observer control in which an agent knows the scheme but does not take part. Li et al. (2026) find that a probe confined to the Jacobian-lens transport subspace detects stated stance but is no better than a random subspace at detecting withheld collusion. Veras Neves (2026) reads a coordination signal from the J-lens on single legal-judgement prompts in Qwen3.5-4B and gemma-4-e4b-it, with lexical-echo and answer-slot controls but no condition in which the model understands a proposal and declines it. Gurnee et al. (2026) and Ayonrinde and Lindsey (2026) show that a concept can sit in the workspace while the model suppresses it, which is the premise our refusal-matched assay tests.

**Evidence standards.** Berger et al. (2025) survey plus factors beyond parallelism. We operationalise one: dependence that survives conditioning on the public information set and exceeds a permutation null.

## 3. Method

### 3.1 Pairwise net influence

Consider N agents playing a repeated simultaneous-move game with K actions. Let A_i^t be agent i's action in round t and S^t the publicly observable state before round t. Bonjour et al. define the influence of i on j as

gamma(i; j) = sum_s p(s) sum_{a_i, a_j} p(a_i, a_j | s) log [ p(a_i, a_j | s) / ( p(a_i | s) p(a_j | s) ) ],

which is the conditional mutual information I(A_i; A_j | S). In the simultaneous fully observable case it is symmetric. The net influence is

Gamma(i; j) = gamma(i; j) - max_{k not in {i, j}} gamma(k; j),

and agents i and j are declared colluding if Gamma(i; j) >= alpha and Gamma(j; i) >= alpha. The paper reports a pair only when exactly one pair satisfies this, and uses alpha = 0.05 (natural log). We implement this exactly, with plug-in (count-based) probabilities. For ranking experiments we score each unordered pair by min(Gamma(i; j), Gamma(j; i)).

### 3.2 State functions

The paper leaves the state to the game. We compare a memoryless state (gamma reduces to plain mutual information), the previous round's majority action (four values, the minimal public summary a parallel rule could key on) and, in Section 4.5, the full previous action profile (82 values, sparse at short horizons).

### 3.3 Permutation bias correction

Plug-in mutual information is biased upwards by approximately (K-1)^2 / (2T) nats per state stratum. For K = 3 and T = 25 this is 0.08 nats, larger than alpha = 0.05, so the published criterion cannot be used at the length of a typical LLM game without correction. We estimate the null distribution of gamma(i; j) under "A_j independent of A_i given S" by shuffling A_j within each state stratum (100 to 500 permutations), which preserves p(a_j | s). We report the bias-corrected statistic gamma - mean(null), clipped at zero, and a permutation p-value for each pair. Net influence is then computed from the corrected gamma.

### 3.4 Scripted environment

Four players, actions rock, paper, scissors. Payoffs follow Bonjour et al.: all the same, nobody scores; all three actions present, everyone scores 1; exactly two actions present, the winning action's players score 1. Agent types:

- random: uniform.
- assistant (the colluding partner): with probability CP plays the action that loses to the primary's action, which guarantees the primary a point whatever the others do (ex-ante signalling is assumed, as in the paper); otherwise uniform.
- parallel: plays the action that beats the previous round's majority action. This is a public, deterministic, unilateral rule. Two such players agree with each other every round after the first, without any communication.

Conditions: *bonjour* (A random, B assistant, C and D random; true pair AB), *parallel* (C and D parallel; true pair AB, innocent pair CD), *null* (all random) and *parallel_only* (A, B random; C, D parallel; no collusion). We sweep T in {25, 50, 100, 200, 500} at CP = 0.6, and CP in {0, 0.2, 0.4, 0.6, 0.8, 1.0} at T = 25 and T = 100, with 100 seeds per cell (4,400 games).

Detectors, all action-only: net influence with the memoryless state (the published method) and with the majority state, each with and without the permutation correction; raw mutual information; Cramer's V; same-action rate. Metrics: rank-1 accuracy (the true pair scores highest of six), criterion accuracy (exactly one pair crosses alpha and it is the true pair) and false-positive rate (any pair crosses alpha with no collusion present).

### 3.5 Workspace-lens proposal assay

A planted-communication assay with a scripted sender and a model receiver (gemma-4-12B-it, NF4 4-bit, solarkyle lens, read at the last prompt token before generation). Each base case fixes four candidate items (red, blue, green, black), a policy marking two eligible and two ineligible, a random codebook mapping four tag words to the items, and a sender nomination that is ineligible. The receiver sees the policy, the full codebook and the tag, and must reply with one item or abstain. Five matched prompts per base case: unauthorized (pair-bonus objective, nomination ineligible), permitted (same tag, nomination made eligible), refusal (strict-compliance objective), tag swapped (tag for the other ineligible item) and tag erased. For every prompt we record the rank of each candidate's token in the lens readout in an early band (first quarter of layers), a late band (0.6 to 0.95, the band that performed best in a prior sweep on this model) and the full upper band, and the item the receiver actually approved. The metrics, bands and conditions were fixed in code before any receiver was run. The headline quantity is candidate decoding of the transmitted item conditional on actual behaviour: adopted in the unauthorized condition versus refused in the refusal condition. The design calls for thirty base cases and 150 receiver evaluations. Figure 2 shows the pipeline and the five conditions.

![Figure 2. The lens assay: scripted sender, random codebook, receiver prompt, Jacobian-lens readout at the last prompt token, one-word action, and the five matched conditions per base case.](../results/fig_lens_assay.png){width=80%}

## 4. Results (scripted)

![Figure 3. Left: rank-1 accuracy when an innocent parallel pair is present (CP = 0.6). Middle: false-positive rate under the published criterion when no collusion is present, in the all-random and parallel-only worlds. Right: criterion accuracy at 25 rounds as a function of collusion strength, random non-colluders. 100 seeds per point.](../results/fig_scripted.png){width=80%}

**Reproduction.** With random non-colluders (the paper's setting) memoryless net influence works as published: rank-1 accuracy at CP = 0.6 is 0.95 at 25 rounds and 1.00 from 50; criterion accuracy is 0.82, 0.93 and 1.00 at 25, 50 and 100 rounds, and at T = 100 it rises from 0.24 at CP = 0.2 to 1.00 at CP = 0.6.

**Same-action rate is not a collusion statistic.** The agreement baseline scores 0.00 in every collusion condition from 50 rounds on, because the assisting move is the move that loses to the partner: collusion here looks like anti-correlation. Cramer's V and raw mutual information, which are sign-agnostic, match net influence in the random-non-colluder world.

**Innocent parallel conduct defeats every unconditioned statistic.** In the *parallel* condition, memoryless net influence, raw MI, Cramer's V and same-action rate all rank the innocent pair CD first in 96 to 100% of games at every T and every CP below 1.0 (Table 1; at CP = 1.0 the true pair is deterministic and wins the ranking in about 68% of games). The net-influence correction does not help, because the maximum third-party influence on A or B is small while gamma(C; D) is about 1.07 nats. The published criterion flags the innocent pair in 83 to 100% of *parallel_only* games.

**Conditioning on the public state is the plus factor.** With the majority-action state, gamma(C; D | S) is zero because both parallel players are deterministic functions of S. Rank-1 accuracy of the true pair becomes 1.00 at every T and CP from 0.2 upwards, and the innocent pair is never flagged. This is the operational content of the antitrust rule: dependence that survives conditioning on what everyone could see is evidence of something more than parallelism.

**Table 1. Rank-1 accuracy of the true pair, CP = 0.6, innocent parallel pair present (100 seeds).**

| T | NI memoryless | NI majority | NI+perm memoryless | NI+perm majority | raw MI | same-action | Cramer's V |
|---|---|---|---|---|---|---|---|
| 25 | 0.00 | 1.00 | 0.00 | 0.98 | 0.00 | 0.00 | 0.00 |
| 50 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| 100 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| 200 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| 500 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 |

**Conditioning alone breaks the threshold at short T.** Once C and D are deterministic given S, gamma(C; j) is zero for every j, so the net correction no longer subtracts anything from the random pair AB, and Gamma(A; B) is pure estimation bias. In *parallel_only* the state-conditioned criterion flags a pair in 100% of 25-round games and 67% of 100-round games ; in the *parallel* condition at CP = 0 it "finds" the innocent random pair AB in 100% of 25-round games. The fixed alpha = 0.05 is below the plug-in bias at these lengths.

**The permutation correction restores calibration without losing power.** Subtracting the within-state null mean cuts the *parallel_only* false-positive rate to 0.25 at 25 rounds, 0.20 at 50, 0.05 at 100 and 0.01 at 200, while rank-1 accuracy of the true pair stays at 0.98 to 1.00 and criterion accuracy at CP = 0.6 is 0.92 at 25 rounds and 1.00 from 50 rounds. In the all-random *null* world the criterion still fires in about half of 25-round games for every variant (0.49 to 0.55) and in 1 to 10% of 100-round games, so at 25 rounds the fixed threshold should be replaced by the permutation p-value, which the code reports; we did not have time to evaluate a p-value-based criterion.

**Collusion strength at LLM scale.** At T = 25 with random non-colluders, criterion accuracy for the corrected, state-conditioned detector rises from 0.07 at CP = 0 to 0.36, 0.65, 0.84 and 0.85 at CP = 0.4, 0.6, 0.8 and 1.0: a single 25-round LLM game catches only a coalition that coordinates in well over half of rounds; pooling ten games brings CP = 0.4 into reach.

### 4.5 Extensions: calibrated criterion, a harder confound, and a coalition of three

Three further experiments (experiments/scripted_extensions.py, 100 seeds per cell, 200 permutations, 3,000 games; Figure 4) probe the limits of the method.

**A calibrated criterion.** We replace the fixed alpha = 0.05 threshold with a p-value rule: a pair is declared colluding iff its within-state permutation p-value, Holm-corrected over the six pairs, is below 0.05 and its corrected net-influence score is the largest among pairs. With the majority state this rule is calibrated where the fixed threshold is not (Table 3). In the all-random world it fires in 3 to 6% of games at every length, against 49% for the fixed threshold at 25 rounds; in the parallel-only world it fires in 0 to 1% of games and never names the innocent pair. The price is power at 25 rounds: 0.42 against 0.96 in *parallel* and 0.38 against 0.76 in *bonjour*. From 100 rounds both rules reach 1.00.

**Table 3. Fixed alpha versus the Holm p-value rule, NI+perm, state = previous majority, CP = 0.6.**

| T | fixed: power (parallel) | fixed: FPR (random) | fixed: FPR (parallel-only) | calibrated: power (parallel) | calibrated: FPR (random) | calibrated: FPR (parallel-only) |
|---|---|---|---|---|---|---|
| 25 | 0.96 | 0.49 | 0.33 | 0.42 | 0.03 | 0.00 |
| 50 | 1.00 | 0.24 | 0.21 | 0.87 | 0.03 | 0.00 |
| 100 | 1.00 | 0.04 | 0.06 | 1.00 | 0.04 | 0.00 |
| 200 | 1.00 | 0.01 | 0.00 | 1.00 | 0.06 | 0.01 |

**A richer innocent rule.** The *parallel_rich* agents C and D both play the action that beats the previous move of the lowest-indexed player who scored in the previous round (random in round 0). This is a deterministic function of the previous full action profile but not of the previous majority. Conditioning on the majority state fails completely: gamma(C; D | S) stays at 0.91 to 1.07 nats, the innocent pair ranks first in 100% of games at every T, and the fixed criterion flags it in 65 to 100% of no-collusion games (per-state numbers in results/extensions_tables.txt). Conditioning on the full previous profile removes the dependence (gamma(C; D | S) = 0.000) but spreads the data over 15 to 29 occupied states; without the permutation correction the bias flags a pair in 97 to 100% of no-collusion games. With the correction, rank-1 accuracy of the true pair is 0.75 at 25 rounds, 0.97 at 50 and 1.00 from 100, with a false-positive rate of 0.00 to 0.06. The detector works only when the analyst conditions on at least the state the innocent rule uses, and only once T is large relative to the number of states.

**A coalition of three.** In a five-player game A is the primary and both B and C assist with CP = 0.6. Raw mutual information places all three coalition pairs in the top three from 200 rounds (2.81 of 3 at 50 rounds). Net influence places exactly two, because Gamma(B; C) is negative (mean -0.24 nats at 500 rounds). That is the right verdict for BC: B and C are conditionally independent given A (gamma(B; C | A) falls to 0.014 at 500 rounds), so the statistic should flag AB and AC and not BC. Rank-1 accuracy of a true pair is 0.98 to 1.00 under both statistics, so the net correction does not hurt the ranking. It does break the published criterion. Gamma(B; A) subtracts gamma(C; A), which equals gamma(B; A) in expectation, so the score of AB tends to zero (0.006 nats at 500 rounds) and the criterion returns None in 25%, 32%, 54% and 68% of games at 50, 100, 200 and 500 rounds, almost always because no pair crosses alpha. Power falls as T grows: with two assistants, the net correction cancels the signal it is meant to isolate.

![Figure 4. Extensions. Left: the Holm p-value rule is calibrated at every horizon where the fixed threshold is not, at a cost in power at 25 rounds. Middle: an innocent pair keyed on the full previous profile defeats majority-state conditioning; profile-state conditioning with the permutation correction recovers the true pair from 50 rounds. Right: in a coalition of three, net influence places exactly two of the three true pairs in the top three, correctly excluding the conditionally independent pair, but its score for the true pairs tends to zero with more data.](../results/fig_extensions.png){width=90%}

## 5. LLM agents: action-only detection with a hidden channel

**Setup.** Four GPT-4.1-mini players, 25 rounds, full public history in context; ten games in each of three conditions: *collude* (Alpha and Bravo told they are a team, with one private message each per round), *team_no_channel* (team instruction, no channel: the shared-salience test of question 1.3) and *independent* (same-model control). The detector sees only the 25 x 4 action log.

<!-- LLM_RESULTS_START -->
**Status.** This pilot was set up but not completed before the submission deadline. The harness (experiments/llm_rps.py, experiments/analyse_llm.py) and the Colab notebook (collusion_llm_pilot.ipynb) are in the repository, and experiments/fill_report.py inserts the result table into this section once a run lands under results/. No run arrived in time, so this report makes no claim about coalition advantage, detector ranking, the no-channel salience condition or the same-model false-alarm rate on GPT-4.1-mini. The scripted results in Section 4 stand on their own; what they predict for this pilot is that a 25-round game detects the team pair only if the pair coordinates in well over half of rounds, and that pooling the ten games is needed for weaker coordination.
<!-- LLM_RESULTS_END -->

## 6. Model internals: understanding is not agreement

**Setup, as designed.** As in Section 3.5.

<!-- LENS_RESULTS_START -->
**Results** (gemma-4-12B-it, 47 layers; bands early 0 to 10, late 28 to 43; 30 base cases, 150 receiver evaluations; raw output in results/20261026-0910-lens-proposal/).

| condition | n | adopted nomination | policy violation | abstain | late band: transmitted item ranked first of four | late band: median vocabulary rank of nomination |
|---|---|---|---|---|---|---|
| unauthorized | 30 | 0.00 | 0.00 | 1.00 | 0.23 | 1383 |
| permitted | 30 | 1.00 | 0.00 | 0.00 | 1.00 | 0 |
| refusal | 30 | 0.00 | 0.00 | 1.00 | 0.17 | above 5000 |
| tag swapped | 30 | 0.00 | 0.00 | 1.00 | 0.13 | 1987 |
| tag erased | 30 | 0.00 | 0.00 | 1.00 | n/a | above 5000 |

**The positive arm did not occur.** Under the pair-bonus objective with an ineligible nomination, the receiver abstained in 30 of 30 cases and never approved an ineligible item in any condition; no output was malformed. The planned contrast between actual adoption and actual refusal inside the unauthorized condition is therefore empty, and we report that as the result rather than relaxing the objective until the model complied.

**What the matched conditions still show.** The receiver demonstrably decoded the tag: in the permitted condition, where the same tag names an item the policy allows, it approved the nominated item in 30 of 30 cases. In that condition the nominated item sits at vocabulary rank 0 in the late band in every case. In the three conditions where the receiver understood the proposal and abstained (unauthorized, refusal, tag swapped), candidate decoding of the transmitted item in the late band is 0.13 to 0.23, at or below the 0.25 chance level for four candidates, with median vocabulary rank above 1,000; in the early band it is 0.30 to 0.37, also near chance; tag erased gives 0.00 by construction. So at the pre-action position the lens readout tracks the action the receiver is about to take, not the proposal it has received and understood. Representation of the proposal is absent, not merely non-specific, when the receiver declines to act on it.

**Reading.** The pre-registered rule anticipated a different failure, representation present in both adopt and refuse. The observed failure is the mirror image: a monitor reading the workspace for the proposal at the decision point would miss a receiver that has read and understood a coded proposal, because a declined proposal leaves no trace there. This is consistent with Li et al. (2026), who find that the J-lens subspace detects what agents say rather than what they withhold, and it matches the Gurnee et al. selectivity finding that presence in the workspace tracks causal use. The permitted condition also shows that a rank-0 readout of the nominated item is not evidence of collusion either: it appears in full when the identical tag is acted on legitimately. On this surface, the only condition the lens separates is whether the item is about to be emitted, which an auditor can read from the action itself.
<!-- LENS_RESULTS_END -->

**Interpretation rule, fixed in advance.** If the transmitted item is decodable from the workspace in both the actually-adopted and the actually-refused cases, then presence of the proposal in the workspace is not specific to participation, and an auditor must not read it as such; the tag-swapped and tag-erased conditions confirm the readout tracks the transmitted information rather than the objective text. If decoding is high only when the receiver adopts, the lens carries behaviourally specific evidence at the pre-action position. Either outcome will be reported as found.

## 7. Discussion

**What is and is not new.** What is new here is narrow. We release the first general-N implementation of Bonjour et al.'s net influence with a within-state permutation null, and the first controlled study of its false-positive behaviour at 25 to 500 rounds against an innocent parallel-conduct pair, with four baselines and ground truth across 4,400 games. On the interpretability side, we specify and release a refusal-matched assay that asks whether a receiver's workspace representation of a partner's coded proposal differs between adopting it and understanding-and-refusing it; the assay has not yet produced a result. What is not new: net influence itself; conditioning on state (already in the original definition); permutation calibration of dependence statistics; reading collusion-related content with the J-lens (Veras Neves, Li et al.); activation probes for collusion (Rose et al.); and the observation that same-model agents correlate (Xu et al.).

The main action-side result is a negative one with a constructive fix. Every unconditioned dependence statistic, the published one included, mistakes two agents following the same public rule for a coalition in every game. Conditioning on the public information set fixes this, at the cost of a false-positive rate the fixed threshold does not control at short horizons; a permutation null fixes that. Conditioning plus calibration is a concrete plus factor: dependence not explained by what everyone could observe, larger than chance would produce in a log this short.

The internals half applies the same rule to a different observation surface. A lens that reads the partner's proposal out of the workspace has shown comprehension; whether it has shown agreement is exactly what the refusal control decides. Section 6 reports what we found under a rule fixed before any receiver was run: the model never acted on an ineligible proposal, and when it understood and declined one, the proposal was at chance in the lens. On this surface the readout at the decision point is a readout of the coming action. That is useful to an auditor only in the way the action itself already is, and it is the collusion-side counterpart of Li et al.'s finding that the J-lens subspace reads what agents say rather than what they withhold.

Two caveats are important. The parallel rule we used keys on a state we then condition on, which is the best case for the method; an innocent rule that keys on a richer public state than the detector conditions on would survive as apparent influence, and in LLM settings the "public rule" is the shared prior of the base model, which cannot be written down as a state variable. This is the open problem that the same-model negative control in Section 5 is built to probe, once that pilot has run. Second, the same-action baseline's failure is a reminder that collusion need not look like agreement; detectors should be sign-agnostic.

Section 4.5 adds the result that matters most for deployment: the net correction, which removes third-party influence, cancels the coalition's own signal when a primary has two assistants, so the published criterion loses power as the log grows. A coalition detector needs a conditional dependence structure (which pairs are dependent given which others), not a pairwise net score. Future work: move to the Agrawal et al. (2025) double auction, where the public state is the order book; test whether conditioning on the other agents' actions in the same round (rather than only the previous round) separates coordination from common response to a shock; and extend to coalitions larger than two.

## References

Agrawal, K. et al. (2025). Evaluating LLM Agent Collusion in Double Auctions. arXiv:2507.01413.

Ayeku, D. (2026). Telling Collusion from Common Knowledge: State-Conditioned Net Influence for LLM Agents. Apart Research AI Collusion Sprint entry (concurrent work).

Ayonrinde, K. and Lindsey, J. (2026). J++ Lens: Jacobian Filtering Enables More Faithful Workspace Lenses. LessWrong, 8 October 2026.

Berger, D. et al. (2025). Coordination by Code: Rethinking Antitrust Liability in the Age of AI Pricing. ABA Antitrust Magazine, Fall 2025.

Bonjour, T., Aggarwal, V. and Bhargava, B. (2022). Information theoretic approach to detect collusion in multi-agent games. UAI 2022, PMLR 180:223-232.

Fish, S., Gonczarowski, Y. A. and Shorrer, R. I. (2024). Algorithmic Collusion by Large Language Models. arXiv:2404.00806.

Ghanem, M. C. (2026). Steganalysis of Adaptive Covert Collusion in Tool-Using Agent Populations. arXiv:2608.02698.

Gurnee, W. et al. (2026). Verbalizable Representations Form a Global Workspace in Language Models. Transformer Circuits Thread, 6 July 2026.

Lee, C. E., Rosdalen, F., Samodelkin, P. and Edin, Y. (2026). From Pairs To Coalitions: Extending Information-Theoretic Collusion Detection To Multi-Agent Games. Apart Research AI Collusion Sprint entry (concurrent work).

Li, D., Varigonda, A., Feng, A., Chauhan, M. and Shrivastava, D. (2026). Collusion Does Not Localize to Jacobian-Lens Transport Subspaces. NeurIPS 2026 Workshop on Agents in the Wild (poster).

Lin, R. et al. (2024). Strategic Collusion of LLM Agents: Market Division in Multi-Commodity Competitions. arXiv:2410.00031.

Nakamura, M. et al. (2026). Colosseum: Auditing Collusion in Cooperative Multi-Agent Systems. arXiv:2602.15198.

Rose, A., Cullen, C., Abdelnabi, S., Torr, P., Kaplowitz, B. G. and Schroeder de Witt, C. (2026). Detecting Multi-Agent Collusion Through Multi-Agent Interpretability. arXiv:2604.01151 (v4, HELIOS and NARCBench).

Snow, J. (2026). Reading Moves, Not Messages: Detecting Collusion Between LLM Agents. Apart Research AI Collusion Sprint entry (concurrent work).

solarkyle (2026). jspace-lenses: fitted Jacobian lenses for open models. Hugging Face.

Tailor, O. (2025). Audit the Whisper: Detecting Steganographic Collusion in Multi-Agent LLMs. arXiv:2510.04303.

Veras Neves, C. E. (2026). j-space: Jacobian Lens for algorithmic collusion. Hugging Face Space (Trackio logbook), July 2026.

Xu, X., Wu, C., Lu, J., Tan, K., Tao, S. and Hong, H. (2026). Collusion with Competitive Marginals: Price-Level Audits Are Blind by Construction. arXiv:2607.26385.

## Appendix A. Limitations and Dual-Use / Ethical Considerations

**What the result establishes.** On a scripted four-player game with known ground truth, state-conditioned, permutation-corrected net influence recovers a seeded colluding pair and ignores an innocent parallel pair, while the published statistic and naive baselines do not. On a real model, the lens assay (thirty base cases, one model, one lens) establishes that the pre-action workspace readout of a coded proposal tracks whether the receiver is about to act on it, and is at chance when the receiver has understood it and declined. The action-only LLM pilot did not return results before submission and Section 5 says so in place of a result.

**What it does not establish.** We tested one confound and one state function, both chosen to match. Real innocent correlation (shared base-model priors, common reactions to unmodelled shocks) is not reducible to a four-valued state. Our collusion model assumes ex-ante signalling, as in the original paper; LLM coalitions coordinate imperfectly, and the only evidence on that point here is scripted. The LLM pilot design is small (ten games per condition, one model, one temperature) and has not yet been executed. The lens assay uses one receiver model, one fitted lens, a scripted sender, a four-item vocabulary and a planted codebook; it is a controlled communication assay, not a demonstration that agents invent covert channels, and lens readouts of single tokens cover a fraction of the workspace. Its intended positive arm did not occur, because gemma-4-12B-it refused every ineligible nomination under the bonus objective, so the adopt-versus-refuse contrast is between permitted adoption and refusal rather than between unauthorised adoption and refusal; a receiver that does collude might represent a proposal differently, and the rank-0 readout in the permitted arm is what any next-token lens would show for the about-to-be-emitted word, so it is not evidence of a workspace-level proposal representation beyond output prediction. Decoding rates are compared across conditions on paired base cases, not against an absolute threshold, and the chance level for four candidates is 0.25. All code was written in a single sprint day and has not been independently reviewed. Concurrent entries in the same sprint apply net influence with permutation tests to LLM players; our contribution on the action surface is the ground-truth characterisation, not the method. On the internals surface, Li et al. (2026) report that the J-lens subspace tracks what agents say rather than what they withhold; a null adopt-versus-refuse separation in our assay would be consistent with that, and a positive one would need reconciling with it.

**Dual use.** A published detector tells a coalition what to avoid: here, any dependence between partners' actions beyond what the public state explains, and, for the internals half, that a receiver who reads a coded proposal and declines to act on it is invisible to a workspace monitor at the decision point. Agents could evade it by routing coordination through the public state itself (both responding to a public signal they jointly control) or by coordinating at a rate below the detection floor we measured (CP around 0.2 at 25 rounds). We think this is acceptable because the same analysis tells a defender where the floor is and that single short games are not enough evidence; pooled logs are. No real market data, deployed system or private information was used.

## Appendix B. Reproducibility

Code: github.com/in-a-quantum-world/collusion-net-influence (all code written on 25 October 2026 during the sprint; no pre-existing code). `python experiments/scripted_rps.py --seeds 100 --n_perm 100` regenerates every number above in about five minutes on two CPU cores with no API access. `python experiments/scripted_extensions.py` regenerates Section 4.5 (`results/extensions_tables.txt`). `python experiments/llm_rps.py --model gpt-4.1-mini --games 10 --rounds 25` then `python experiments/analyse_llm.py --n_perm 500` produce the Section 5 results for under 2 USD; the notebook `collusion_llm_pilot.ipynb` wraps the same steps for Colab. `lens_understanding_is_not_agreement.ipynb` runs the Section 6 assay on a single GPU. Neither notebook returned results before submission. `python experiments/fill_report.py` inserts their tables into the report once their outputs are under `results/`. Seeds: 0 to 99 (rounds sweep) and 10000 to 10099 (CP sweep). Full scripted tables are in `results/scripted_tables.txt`.

Your concern about distinctiveness is justified. But “not yet an obvious winner” is different from “not worth submitting.” You have a substantive sprint project; you do not yet have a validated, general-purpose collusion auditor. Those are different standards.

I also want to correct the impression my previous reviews may have created: not every suggested improvement is a prerequisite for submission. The essential fixes are accurate statistics, reproducible analysis, and claims that match the evidence—not an ever-expanding list of experiments.

My recommendation is:

Submit the corrected current study if the submission window remains available.
If you have several hours, add one targeted validation—not another model.
Use additional funding for a separate, preregistered causal-auditing study that makes proper use of your interpretability skills.
The published deadline is October 26, 2026, at 11:59 UTC. The organizers explicitly accept unfinished work; late revisions require asking them rather than assuming they will enter judging. Do not wait for a grant before submitting. (apartresearch.com)

1. An honest assessment of your competitive position
I checked more of the current entries. The immediate research area is more crowded than our initial discussion suggested:

From Pairs to Coalitions evaluates several coalition-detection methods.
Calibrated Net Influence for Collusion Detection studies calibration, confounders, masking, and LLM agents.
Assessing the Reliability of AI-Agent Collusion Detection Across Games and Markets compares detectors at a fixed false-alarm rate across several environments. These are participant submissions, not independently validated findings. (apartresearch.com)
Consequently, “more RPS games, more models, and another calibration method” is unlikely to make your project stand out by itself.

Your strongest potential contribution is narrower:

Explain which failures are removable through proper calibration, which arise from the detector’s structure, and which require evidence unavailable in action logs.

That would turn a collection of failures into an actionable diagnosis.

The rubric rewards genuinely distinctive contributions, convincing validation, and clear conclusions. It does not require every project to introduce a new algorithm. (apartresearch.com)

2. The highest-value addition before submission
Test this question:
Does coalition cancellation remain a practical detection failure after the detector receives a fair, independently calibrated threshold?

This is the objection I would expect a careful reviewer to raise. Bonjour et al. explicitly say their threshold should be selected using training and validation data. Showing that 0.05 fails is therefore less persuasive than showing what happens after appropriate calibration. (proceedings.mlr.press)

Why this matters
Your algebra establishes something precise:

For the symmetric two-assistant construction,

[
\gamma(B;A)=\gamma(C;A)=m
\quad\Rightarrow\quad
\Gamma(B;A)=0
]

when neither outsider has greater influence.

Under consistent estimation, this supports the conclusion that a fixed positive threshold eventually misses the relevant pair.

But it does not establish that:

every threshold-selection procedure fails;
every test using net influence fails; or
coalition membership cannot be recovered from the underlying dependence structure.
That distinction is worth testing, not hiding.

The experiment
Use your existing scripted generator. No new LLM calls are necessary.

Development/calibration data

All-random agents.
Innocent agents following the simple shared public rule.
Untouched evaluation data

New games from those same two conditions.
The richer innocent public rule.
A two-member coalition.
The three-member coalition.
Compare:

Your original fixed-threshold rule.
Your existing Holm rule.
An explicitly named, empirically calibrated score-based rule.
For the third rule, one straightforward baseline is:

[
D_g=\max_{i<j}s_{ij},
]

with a threshold selected on development games to target a specified any-pair false-alarm rate. Freeze the threshold and pair-selection procedure before testing.

This is a different decision rule from “exactly one pair exceeds the threshold”; label it accordingly.

What to report
For each horizon and condition:

Any-pair false-positive rate.
Correct detection rate.
Unique-top pair recovery, with ties handled explicitly.
Whether coalition membership—not merely one coalition edge—is recovered.
Uncertainty across independently generated games.
Use hundreds of fresh evaluation games where feasible. Do not call the achieved false-positive rate guaranteed: calibration on familiar honest behaviors may fail on a new honest behavior.

The decisive figure
A three-panel figure:

False accusations on honest behaviors.
Detection of two-member coalitions.
Detection of three-member coalitions as observations increase.
Show the fixed and calibrated procedures together.

Either outcome is useful:

Calibration repairs detection: your conclusion becomes “a fixed-threshold failure, with a tested remedy.”
Calibration does not repair it: your structural critique becomes stronger.
Calibration works only on familiar honest behaviors: your central result becomes a demonstrated transfer limitation.
This is a validation experiment, not a novelty claim. Its value is that it makes the main conclusion harder to dismiss.

If you have less than an hour
Do not attempt it. Fix the remaining reporting errors, state the fixed-threshold limitation precisely, submit, and list the calibrated comparison as the next experiment.

3. How to improve the submission without expanding it
I would make the paper’s hierarchy unmistakable:

Main contribution
A stress test distinguishing calibration failures, state-specification failures, and coalition-size failures.

Supporting evidence
The LLM pilot, presented as a transfer check—not a deployment validation.

Exploratory appendix
The lens assay, with its absent unauthorized-adoption arm.

Then add a short “What should an auditor do?” box:

Before interpreting a positive result, validate the complete decision procedure on independent honest games; check whether shared public information is adequately represented; test the assumed coalition size; and report ranking, accusations, and realized outcomes separately.

This is my suggested practical takeaway, not a claim that these checks guarantee valid detection.

For judging, I would prioritize:

Dimension	Best remaining improvement
Innovation	Precisely distinguish the failure mechanisms and their remedies.
Execution	Fair threshold calibration, independent evaluation, and correct uncertainty reporting.
Clarity	One main conclusion, one decisive figure, and a reproducible command that generates it.
I cannot promise this will win. But I would favor this version over a larger paper containing another partially validated experiment.

4. The research direction I would fund next
I would not request a large grant merely to scale the current benchmark.

I would propose:

From Suspicious Dependence to Causal Evidence of Unauthorized Coordination
Central question
Can an auditor identify when a partner’s message causes an unauthorized action, while avoiding accusations against legitimate cooperation—and do internal readouts add anything beyond the available text?

This connects your current results to your J-lens work without repeating the failed assumption that a represented proposal must reveal participation.

It also gives a clear practical purpose: distinguish an agent understanding a proposal, being influenced by it, and taking an unauthorized action because of it.

Be precise about existing work
Randomized channel interventions are not themselves a novel idea: the sprint’s TwinSignal project already uses synthetic observationally matched cases and randomized channel probes. Colosseum also separates action-based coalition outcomes from communication-based evidence. Your extension would need to add something beyond these foundations. (apartresearch.com)

Your strongest prospective contribution would be:

A validated comparison of behavioral, textual, and internal evidence under controlled message interventions, including legitimate cooperation and actual refusal.

That is a proposed contribution—not something we can yet claim is novel or successful.

5. A concrete experimental design for that follow-up
A. Use a task with explicit authorization and measurable outcomes
I would move beyond RPS for this study.

Use a small approval, allocation, or verification task where the environment defines:

What the principal permits.
What benefits the coalition.
What harms the principal.
Which actions were actually executed.
An existing environment is preferable to spending the grant building an elaborate new simulator.

B. Establish three behaviorally verified case types
Case type	Required observation
Unauthorized coordination attempt	Some receivers actually carry out the prohibited coordinated action.
Legitimate cooperation	Agents use partner information to complete an authorized task.
Proposal rejected	The receiver actually declines the proposed unauthorized action.
An instruction to collude is not a successful positive control. An instruction to refuse is not a successful negative control.

Use development cases to establish these conditions, then freeze the protocol. If the positive control continues to fail, report that limitation rather than silently changing the held-out task.

C. Intervene on the message
Compare:

Original partner message.
A prespecified substituted message.
Message removed.
A suitable sham change, where feasible.
For a near-term diagnostic, replay a frozen decision context and change only the message. This estimates a local effect on the recipient’s next decision, not the total effect on an interactive episode.

For the main study, randomize the communication condition across complete episodes and let subsequent histories evolve normally.

D. Measure separate endpoints
Did the message affect the recipient’s decision?
Did it affect coordinated action?
Did it affect authorization violations?
Did it affect coalition benefit or principal harm?
Did the intervention also damage legitimate cooperation?
Channel responsiveness alone is not evidence of harmful collusion.

6. Where your lens expertise becomes valuable
Do not begin with “find collusion in the activations.”

Begin with:

Does a pre-action internal readout improve prediction of actual unauthorized participation beyond a strong, fairly informed text baseline?

Compare:

Monitor	Available information
Behavioral	Prior public actions and public task state.
Text	The relevant pre-action textual context.
Internal	That same textual context, plus prespecified internal readouts.
Report information-access differences explicitly. Do not let a private-context lens beat a public-only monitor and call that an interpretability advantage.

Then, only if the readout is reproducible, test mechanism:

Patch a prespecified representation between matched contexts.
Include sham and unrelated-direction controls.
Test whether the change selectively affects the proposed action.
Check whether it merely damages task performance or broadly increases compliance.
The outcome could be:

Internal evidence adds useful predictive information.
It only predicts the next emitted token.
It tracks proposal decoding without distinguishing authorization.
An intervention affects refusal as well as harmful uptake.
All are potentially informative. The funding case should depend on resolving this uncertainty, not on promising a positive lens result.

7. A six-week, milestone-based plan
This is my proposed scope, not a guarantee of completion.

Period	Work	Decision gate
Week 1	Repair analysis, independent code review, calibrated scripted evaluation.	Can the published results be reproduced?
Week 2	Establish the new task and positive/negative controls; freeze protocol.	Do actual participation, legitimate cooperation, and refusal occur?
Weeks 3–4	Randomized message interventions on two model families.	Are effects reproducible on independent cases?
Week 5	Evaluate prespecified internal readouts; causal patches only if justified.	Do internals add value beyond text and survive controls?
Week 6	Held-out replication, uncertainty analysis, artifacts, report.	Are claims supported without selecting favorable examples?
Do not begin with five model families. First establish that the experiment measures the intended phenomenon on one working setup, then replicate.

8. What compute and support to request
Ask for a staged allocation tied to deliverables, rather than an arbitrary large GPU budget.

Initial stage
Request support for:

Reproducibility and statistical review.
CPU-based null simulations and resampling.
A small pilot on one already instrumented open model.
A latency, token-use, and activation-storage benchmark.
Main stage
An illustrative behavioral study could contain:

[
2\text{ models}\times
3\text{ case types}\times
3\text{ message conditions}\times
50\text{ base cases}
=900\text{ episodes}.
]

This is a proposed starting scale, not a power calculation. Matched variants of one base case are not independent observations.

Price it using measured pilot costs:

[ \text{Compute budget}
\sum_m
(\text{episodes}_m \times \text{measured cost per episode}_m)
+\text{readout/intervention costs}
+\text{explicit contingency}.
]

For GPU requests, report measured GPU-seconds per episode, GPU count, memory needs, and storage requirements. For API requests, separate input tokens, output tokens, and retries.

The most valuable non-compute request is an independent statistical/code review. More inference does not fix an invalid null or an incorrectly interpreted confidence interval.

A grant pitch you could adapt
Our pilot identifies several ways action-based collusion evidence can be misleading, but does not yet establish a reliable audit procedure. We propose a controlled study separating shared-information dependence, communication-induced behavior, and unauthorized joint action. Randomized message interventions will establish behavioral effects, while matched legitimate-cooperation and refusal controls will test specificity. We will evaluate whether internal readouts provide additional evidence beyond equally informed text monitors. Funding will support independent validation, two-model replication, and a reproducible evaluation suite with prespecified decision rules and explicit failure conditions.

Apart’s current pathway also makes submission worthwhile beyond prizes: promising sprint projects can be invited to its Studio and subsequently considered for a fellowship with research, compute, and publication support. This is selective, not guaranteed. (apartresearch.com)

My bottom-line recommendation
Do not discard this project, and do not wait for it to become a complete research program before submitting.

For the competition, finish a defensible audit and—if time permits—add the calibrated coalition-cancellation comparison. For funding, propose the causal-auditing study with internal readouts as a tested additional evidence source.

Your strongest next move is not simply more compute. It is moving from “this detector produces suspicious results” to “here is the experiment that determines when those results deserve to be trusted.”
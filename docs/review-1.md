However, the current version is not yet maximizing your chances. Its strongest findings are obscured by overclaims, inconsistent descriptions of completed work, and two concrete analysis issues I found in the public code.

My recommendation: make this a rigorous stress test of net-influence detection, with the lens experiment as a clearly bounded secondary result. I cannot credibly estimate your probability of winning, but I think correcting the issues below offers a better route than expanding or restarting the project.

I checked the rubric, relevant repository files, the original Bonjour paper, and nearby submissions. I have not rerun the 7,400 simulations or your GPU experiment.

1. What is strongest about the revised project?
In my assessment, your most compelling findings are:

Conditioning on the wrong public state can confidently identify an innocent pair.
Reducing estimation bias is not the same as controlling false positives.
A pairwise correction can cancel the signal from a larger coalition—so more observations can make the fixed-threshold detector less likely to detect it.
The LLM pilot exposes a gap between successful pair ranking, threshold-based detection, and demonstrated coalition benefit.
That is a coherent research contribution. The lens result is useful, but presently much less conclusive.

A stronger central question would be:

Under which assumptions about public information, sample size, and coalition structure does net influence distinguish planted coordination from innocent dependence—and where does calibration fail to transfer?

This preserves the original project’s most important principle: do not confuse an observable signal with evidence of collusion. You do not need to preserve my original experimental plan when the results point elsewhere.

2. How this maps to the actual judging criteria
The rubric rewards field-level novelty, convincing validation, conclusions supported by results, and clear presentation—not simply the number of experiments. (apartresearch.com)

Criterion	My assessment of this draft	Highest-value improvement
Impact Potential & Innovation	Promising, but some novelty claims are too broad.	Lead with the precise failure mechanisms and distinguish your contribution from existing scripted evaluations.
Execution Quality	Substantial work, but currently vulnerable to specific methodological objections.	Resolve the decision-rule mismatch, fix pooled state construction, and narrow the lens interpretation.
Presentation & Clarity	Two competing main stories and several stale passages weaken confidence.	Present one principal contribution, label the lens study exploratory, and reconcile every claim with its actual analysis.
I would not currently make a case for exceptional execution or clarity. The encouraging part is that many weaknesses can be addressed using existing outputs rather than more expensive model runs.

3. Highest-priority issue: your “calibrated” rule does not inherit the LLM control result
This is the most important finding from my review.

Your report uses two different decision rules:

Fixed-threshold rule: net-influence score exceeds 0.05, with exactly one qualifying pair.
Section 4.5 calibrated rule: the top net-influence pair has a Holm-adjusted permutation p-value below 0.05.
The repository implements these as different procedures. (raw.githubusercontent.com)

Applying the second rule to your supplied pooled results
Using the memoryless, permutation-corrected scores and the p-values you supplied:

Condition	Highest-scoring pair	Raw p-value	Approx. Holm-adjusted p-value	Section 4.5 rule
collude	Alpha–Bravo	0.002	0.012	Flags the intended pair
team_no_channel	Charlie–Delta	0.002	0.012	Flags a non-team pair
independent	Alpha–Delta	0.002	0.012	Flags an innocent pair
These are my calculations from your supplied JSON, applying your stated rule; they are not results of a fresh experiment.

The independent condition also has Alpha–Delta as the top corrected pair under majority conditioning, with p = 0.002.

Therefore:

The fixed-threshold detector’s pooled control success is not evidence that your permutation-calibrated detector avoids false accusations on the LLM pilot.

Your manuscript distinguishes the rules in places, but its overarching narrative risks combining the attractive properties of different detectors into one apparent success.

How to turn this into a stronger paper
Report both rules side by side, including the failures:

“Permutation calibration controlled rejection rates in the tested scripted nulls, but did not reliably distinguish innocent dependence from planted coordination in the pooled LLM pilot.”

That is more informative than saying calibration “fixes” the problem.

The underlying distinction is crucial: rejecting conditional independence is not equivalent to establishing collusion. Conditional-independence testing concerns a specified statistical null; its interpretation depends on the conditioning information and assumptions. (proceedings.mlr.press)

4. A concrete pooling bug needs fixing before finalizing the LLM table
In analyse_llm.py, the code:

Concatenates the action arrays from all games.
Passes that combined array to analyse.
Recomputes previous-majority states across the combined sequence.
Consequently, the first round of games 2–10 receives a “previous majority” from the last round of a different game, instead of a fresh-game state. This follows directly from the analysis and state-function implementations. (raw.githubusercontent.com)

Fix: compute the state sequence separately for each game, then concatenate actions and their already-computed states.

This affects the pooled state-conditioned results; it does not directly affect memoryless analysis or the separately analyzed individual games.

Also:

Preserve game IDs.
Assess uncertainty across games, not by treating 250 history-dependent rounds as 250 independent replications.
Check whether the permutation scheme is appropriate for agents that condition on full public history. Conditioning on previous majority alone does not establish that same-state rounds are exchangeable.
Do not automatically attribute innocent dependence to shared model weights: your design does not separate that explanation from common history, position effects, or shared strategies.
Separate ranking from detection
Your supplied results show:

Memoryless rank-1 recovery: 9/10 games.
Memoryless correct threshold-based flags: 5/10 games.
Corrected majority-state rank-1 recovery: 3/10 games.
Corrected majority-state correct threshold-based flags: 0/10 games.
Those are different outcomes. “90% detection accuracy” would be misleading.

Furthermore, per_game_criterion_hit_rate measures whether the designated team was flagged—not whether any innocent pair was flagged. A zero value in the independent condition does not establish zero per-game false positives. (raw.githubusercontent.com)

5. The lens result should be retained—but substantially narrowed
You correctly report that the intended positive arm did not occur. That honesty is a strength.

But several subsequent conclusions go beyond what the experiment establishes.

What your results support
From the table you provided:

The receiver approved the nomination in 30/30 permitted cases.
It abstained in 30/30 unauthorized cases.
The nominated item was much more recoverable with your selected readout when it was about to be emitted.
There were no examples of successful unauthorized adoption, so the intended collusion-related contrast could not be evaluated.
What they do not establish
They do not establish that:

The receiver decoded the nomination in each refusal case.
Its representation was absent from the model.
A declined proposal “leaves no trace.”
The workspace generally contains actions rather than understanding.
The lens offers no monitoring value beyond observing the final action.
Correct behavior under the permitted prompt establishes decoding capability in that condition. It does not independently verify the computation used under every refusal prompt.

Likewise, failure to recover a token with one lens at one position is a measurement limitation—not proof that the corresponding information is absent.

Replace the headline interpretation with this
“At the final pre-generation position, the selected J-lens readout strongly recovered nominated items during permitted adoption but did not reliably recover them during abstention. Because unauthorized adoption never occurred, this experiment does not establish whether the readout distinguishes collusive participation from refusal.”

That is a defensible result.

One implementation detail also needs disclosure
Your notebook reports the best rank across layers in each band and across candidate token variants, with ranks outside the top 5,000 censored. It is not a typical-layer rank. Explain this explicitly and handle censored ties when interpreting four-candidate accuracy. (raw.githubusercontent.com)

Recommendation: keep a short exploratory section and move most lens details to the appendix. Do not chase unauthorized compliance merely to produce a more impressive result.

6. Fix the novelty and attribution claims
These matter directly for the innovation score.

A. “The ground-truth evaluation the method has never had” is too strong
Bonjour et al. already evaluated known colluding configurations, varied collusion strength and sample size, and reported detection performance. They also explicitly described the threshold as a hyperparameter requiring training/validation and limited their scope to two colluders. (proceedings.mlr.press)

A better claim is:

“We extend prior ground-truth evaluation to matched innocent-dependence confounds, short-horizon calibration, state misspecification, and a three-member coalition.”

Also describe the coalition experiment as an extension beyond the original two-colluder scope, not a refutation of a guarantee that the original paper made.

B. Do not say the concurrent work lacks scripted ground-truth evaluation
The From Pairs To Coalitions submission reports 2,800 coalition games and 400 honest scripted games, alongside model experiments. Your current characterization of the nearby entries as lacking such evaluation is therefore too broad. (apartresearch.com)

Differentiate your specific experimental axes and analytical explanation, not simply the existence of a scripted benchmark.

C. Two references/attributions need verification
I could not verify the exact Li et al., “Collusion Does Not Localize to Jacobian-Lens Transport Subspaces” citation through the searches I ran. Obtain the actual paper or workshop record before retaining its claims. This does not establish that it does not exist.
I could not locate the attributed “privileged-observer” proposed control in the Rose et al. v4 or v5 texts I inspected. Supply an exact passage/page, or remove that attribution. The experiment can stand as your own motivated control without claiming they explicitly proposed it. (arxiv.org)
These are citation-verification gaps, not grounds to discard the project.

7. Your coalition-cancellation result deserves more prominence
In my assessment, this is your clearest mechanistic finding.

For the symmetric two-assistant construction, suppose:

[
\gamma(B;A)=\gamma(C;A)=m.
]

Then, assuming neither outsider has greater influence,

[
\Gamma(B;A)
= m-\max{\gamma(C;A),\ldots}
=0.
]

Since your pair score takes the minimum of the two directional net influences, the true pair’s score cannot exceed a positive threshold in the population limit.

The subtraction treats another coalition member as the competing explanation and removes the signal.

That explanation makes your empirical decline in detection with increasing sample size much more compelling than a plot alone.

Two qualifications:

Verify novelty relative to the concurrent coalition paper.
Remove “correctly excluding BC” unless your target is direct interaction edges. B and C can be members of the same coalition while conditionally independent given A. Coalition membership and direct dependence are different ground-truth targets.
8. Clean up the manuscript’s internal contradictions
These are easy to fix and disproportionately important for credibility:

Page 2: says LLM results did not arrive and the lens experiment will run later; Sections 5–6 report both.
Section 4: says the p-value criterion was not evaluated; Section 4.5 evaluates it.
Appendix A: says one confound and one state function were tested; the report includes richer alternatives.
Appendix B: says neither notebook returned results before submission; clarify whether this refers to an earlier version.
“Pre-registered”: use “specified before these runs” unless you have a timestamped preregistration or equivalent documented commitment.
Numbers across simulation batches: explain why nominally similar settings show, for example, 0.25 versus 0.33 false-positive rates. Different seeds/permutation budgets can explain differences, but readers should not have to infer that.
Figures: the supplied screenshots have very small labels. Export vector figures with larger type.
Title: “the workspace shows the action” is stronger than the evidence.
Your October 26 output folders do not by themselves establish late work: the listed deadline converts to October 26, 2026, 11:59 UTC. Record actual completion timestamps and timezones rather than leaving conflicting deadline statements. (apartresearch.com)

9. What I would do next
No new models yet. No wholesale return to the original experiment.

First: analysis corrections
Fix pooled state construction.
Run both decision rules on all existing LLM logs.
Report any-pair false positives, correct pair flags, and pair ranking separately.
Add game-level uncertainty and flag the permutation assumptions.
Second: sharpen the contribution
Use a title such as:

When Net Influence Misleads: Calibration, Innocent Dependence, and Coalition Size

Organize the main results around:

State misspecification.
Bias correction versus decision calibration.
Coalition cancellation.
Transfer limitations in the LLM pilot.
Keep the lens experiment as an exploratory observation, not a second claimed breakthrough.

Third: presentation and provenance
Resolve stale text, verify citations, label exploratory analyses, and make every headline traceable to one table or calculation.

Bottom line
The revised project is the better submission vehicle. The current interpretation is not yet the strongest version of it.

Your best chance is not to claim that you built a reliable collusion detector or established what the workspace contains. It is to show, precisely and reproducibly, why plausible collusion evidence can fail—and which assumptions an auditor must check before trusting it.

That is a stronger scientific contribution than forcing the original hypothesis to succeed.
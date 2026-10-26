This is materially stronger. Keep this title, scope, and overall structure. The revised paper now makes a coherent contribution: it audits the assumptions behind a detector rather than claiming to have solved collusion detection.

You have addressed the biggest problems from the previous draft: separating ranking from detection, exposing the pooled-control failures, explaining coalition cancellation, and limiting the lens claims.

However, I found two remaining uncertainty-reporting errors in the code, plus several important final corrections. These are analysis and presentation fixes—not reasons to restart the project.

I checked the current public analysis script and notebook. I have not independently reproduced the numerical results, and I could not retrieve the analysis JSONs at the paths I tested.

1. Must fix: the benefit interval does not support “not distinguishable from zero”
Your script calculates the 2.5th and 97.5th percentiles of the ten individual game advantages. That describes the spread of observed game outcomes; it is not a confidence interval for the mean advantage. (raw.githubusercontent.com)

Therefore, the reported interval of −3.16 to +5.94 does not justify this abstract claim:

“its benefit is not distinguishable from zero.”

Two acceptable fixes
Fastest—change the wording:

“Mean within-game score advantage was −0.25 in the first run and +1.00 in the second; these descriptive results do not establish a benefit from coordination.”

Label the interval:

“Empirical 2.5th–97.5th percentiles of individual-game advantages.”

Better—calculate uncertainty for the mean: resample whole games and compute the mean advantage in each bootstrap sample, then report the interval of those means.

Also distinguish two questions:

Does the designated pair outperform the other players?
Does enabling coordination cause the pair to perform better than it otherwise would?
Your current within-game advantage measures the first. It does not, by itself, answer the second.

2. Must fix: the score bootstrap is for the uncorrected statistic
The bootstrap loop computes gamma_matrix, then net_influence, without applying the permutation bias correction. The saved output appropriately names this pooled_team_score_boot95_raw_gamma. (raw.githubusercontent.com)

Consequently, the 0.19–0.50 interval should not be presented as uncertainty for the permutation-corrected score.

Either:

Label it explicitly as a game-bootstrap interval for uncorrected net influence, or
Recompute the corrected estimator inside each bootstrap replicate.
For a deadline-constrained submission, relabeling or removing this secondary interval is sufficient. Do not imply that it belongs to a different estimator.

3. Your Holm-budget diagnosis is correct—but fix it if time permits
You correctly identified that with 100 permutations and six pairwise hypotheses:

[
p_{\min,\mathrm{Holm}}=\frac{6}{101}\approx0.0594.
]

Thus, no pair can pass your strict 0.05 criterion, regardless of signal strength.

The notebook still reduces the per-game permutation budget to:

max(100, args.n_perm // 5)
So the advertised --n_perm 500 run still uses only 100 permutations per individual game. (raw.githubusercontent.com)

Recommendation
Use the full permutation budget for individual games and reanalyze the saved logs, not regenerate the agents. For example, use 999 permutations per game.

Your “at least 120” calculation is mathematically correct, but that is merely the minimum at which rejection becomes possible—not a comfortably resolved p-value.

If time is insufficient, your existing disclosure is honest. Keep the cells explicitly unevaluated at adequate resolution, and do not frame their absence as empirical conservatism or low statistical power.

4. Audit ranking ties before retaining the headline accuracy
Your ranking code sorts scores and selects the first pair. Equal scores therefore inherit the enumeration order, which starts with Alpha–Bravo—the designated team. (raw.githubusercontent.com)

This matters because clipping corrected dependence estimates to zero can create ties.

I am not saying your 8/10 or 9/10 results are wrong. I am saying that they require one additional check:

How many top scores were tied?
Did any reported team successes depend on tie-breaking?
If ties occur, report either:

Unique-top recovery, or
Fractional credit: (1/k) when the true pair shares the maximum with (k-1) others.
This is particularly relevant to the independent-control row where Alpha–Bravo ranks first in 5/10 games. In that condition, call it the reference pair, not the “team.”

5. Reanalyze run 1 with the same corrected procedure, if the logs remain available
You now transparently distinguish the two generated datasets. That is good.

But the cleanest comparison is:

Two independently generated datasets, both analyzed with the same final procedure.

Resetting states does not change memoryless raw mutual information. However, changing permutation strata from pooled rounds to within-game rounds can change corrected scores and p-values even in the memoryless analysis.

Therefore, retain the original analysis as provenance, but preferably use the corrected analysis of both runs in the main results.

If the raw run-1 logs are unavailable, say so. Do not regenerate another dataset and call that reanalysis.

6. Four wording changes would make the conclusions substantially safer
A. Replace “a reliable ranking”
Current:

“A single 25-round game therefore gives a reliable ranking…”

Eight successes from ten games, with the interval you report, is not enough to call the ranking reliable.

Use:

“In these pilots, single-game pair ranking was more successful than threshold-based detection, but both estimates remain uncertain.”

B. Replace “the dependence … is real under the permutation null”
You acknowledge that exchangeability is not established, so a small permutation p-value does not settle whether the procedure is statistically valid.

Use:

“The observed associations exceeded those generated by the chosen permutation procedure. Because exchangeability is not established, these p-values are not validated evidence of collusion.”

The failure you demonstrate is the use of this procedure as a collusion decision rule, not necessarily a failure of Holm correction itself.

C. Avoid asserting that no-team controls are certainly innocent
“No planted coalition” is experimentally established. “No collusion of any kind occurred” is stronger, especially where agents observe shared histories.

Use:

“False positives relative to the planted-coalition labels.”

Or simply:

“The detector flagged pairs in controls without a planted coalition.”

This preserves your finding without assuming the conclusion that the detector is meant to adjudicate.

D. Narrow the novelty sentence in the introduction
The statement that the original and concurrent work “do not cover” all four axes is still too broad—your own related-work section acknowledges overlap.

Replace it with:

“We combine these evaluation axes and emphasize their interaction, an analytic account of coalition cancellation, and transfer of two decision rules to the same LLM logs.”

7. The lens section is now appropriately bounded
This is a substantial improvement. You now clearly state:

Unauthorized adoption never occurred.
Permitted decoding does not establish decoding in refusal cases.
One failed readout does not establish representational absence.
Best-rank aggregation and censoring affect interpretation.
I would keep this section secondary and stop expanding it for this submission.

One remaining improvement: Table 5 still contains decoding rates influenced by censored ties. If possible, replace list-order tie-breaking with an explicitly defined unresolved outcome, and report:

Number of evaluable cases;
Unique-target recovery;
Number of unresolved/censored ties.
If you cannot recompute it now, rename the column to “nominal candidate-decoding rate, including tie artifacts.” Your caveat is good, but it should not require readers to discover that the headline percentage contains arbitrary tie decisions.

8. Reference check: one earlier concern is resolved
Your Rose attribution is correct for v1. Section 4.4 explicitly proposes a privileged observer who knows the scheme but is instructed not to participate. My earlier verification was incomplete because I checked later versions. (arxiv.org)

However, match the bibliography’s author list to the version cited: v1 lists Rose, Cullen, Kaplowitz, and Schroeder de Witt, rather than the six-author list currently supplied. (arxiv.org)

I still could not independently verify the Li poster: the NeurIPS page was blocked to my browser. That is not evidence that it is invalid. Before submitting, personally open the actual poster/paper and confirm that it supports your sentence.

9. Small but worthwhile final corrections
Cross-game boundaries: ten concatenated games have nine inter-game boundaries. Unless another issue affected the first game, replace “10 of 250 rounds affected” with “nine boundary rounds.”
Table 4: its caption promises Wilson intervals, but only the ranking column displays them. Add intervals to the detection rates or narrow the caption.
“Four failure mechanisms”: the fourth is an evaluation distinction rather than a separate mechanism. “Four findings” is cleaner.
Independent-control benefit: label it “reference-pair score difference”; there is no coalition there.
Reproducibility: ensure raw game logs—not only printed notebook output—are publicly accessible, and record the final analysis commit.
Figures: enlarge the labels. The supplied plots remain difficult to read at ordinary page scale.
10. Submission-form assessment
Keep:

The project title.
Tracks 1 and 2.
Solo team designation.
Publishing interest: “Yes.”
The eight-page main-text structure.
Your summary is indeed 248 whitespace-separated words. The official requirement is 150–250 words, and references and appendices are excluded from the eight-page cap. (apartresearch.com)

I would nevertheless replace the summary, primarily to remove the unsupported benefit-significance claim.

Suggested replacement summary — 230 words
When does statistical dependence constitute evidence of agent collusion? We stress-test pairwise net influence using 7,400 scripted games and two GPT-4.1-mini pilot runs, each comprising ten games in each of three conditions.

We examine public-state specification, finite-sample calibration, and coalition size. Innocent agents following the same public rule defeat unconditioned dependence statistics; conditioning removes this confound only when the chosen state captures that rule. At 25 rounds, a fixed threshold produces false accusations in approximately half of all-random games. A Holm-corrected permutation rule reduces false-positive rates on the tested scripted nulls, but flags a pair in each pooled LLM control without a planted coalition. These outcomes distinguish statistical rejection from evidence of collusion.

For a symmetric coalition with one primary and two assistants, we derive a cancellation mechanism: the pairwise correction subtracts one coalition member’s influence from another’s. Consequently, fixed-threshold detection deteriorates as sample size increases.

In the LLM pilots, the planted pair ranks first in eight or nine of ten games but is correctly flagged in five or seven. Its mean within-game score advantage changes from −0.25 to +1.00 across runs; these descriptive differences do not establish a causal benefit from coordination.

An exploratory Jacobian-lens assay includes 150 receiver evaluations but no unauthorized adoption, leaving its intended participation-versus-refusal contrast unresolved. We release code and evaluation artifacts and identify assumptions auditors must check before treating dependence as evidence of collusion.

Update its ranking numbers if the tie audit changes them.

Bottom line
This revision is a much more credible competition submission. Do not change direction again.

Prioritize:

Correct the benefit interpretation.
Label the bootstrap estimator accurately.
Check ranking ties.
Reanalyze existing logs with an adequate permutation budget, if time allows.
Finalize the PDF and artifact links.
Those changes strengthen Execution Quality and Presentation & Clarity more directly than another model or experiment would. I would favor this carefully bounded version over the earlier, more ambitious claims—but I still would not promise a win.
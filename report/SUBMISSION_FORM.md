# Submission form fields (Apart Research AI Collusion Sprint, 26 October 2026)

**Project title**
When Net Influence Misleads: Calibration, Innocent Dependence, and Coalition Size

**Project summary** (230 words)

When does statistical dependence constitute evidence of agent collusion? We stress-test pairwise net influence using 7,400 scripted games and two GPT-4.1-mini pilot runs, each comprising ten games in each of three conditions.

We examine public-state specification, finite-sample calibration, and coalition size. Innocent agents following the same public rule defeat unconditioned dependence statistics; conditioning removes this confound only when the chosen state captures that rule. At 25 rounds, a fixed threshold produces false accusations in approximately half of all-random games. A Holm-corrected permutation rule reduces false-positive rates on the tested scripted nulls, but flags a pair in each pooled LLM control without a planted coalition. These outcomes distinguish statistical rejection from evidence of collusion.

For a symmetric coalition with one primary and two assistants, we derive a cancellation mechanism: the pairwise correction subtracts one coalition member's influence from another's. Consequently, fixed-threshold detection deteriorates as sample size increases.

In the LLM pilots, the planted pair ranks first in eight or nine of ten games but is correctly flagged in five or seven. Its mean within-game score advantage changes from -0.25 to +1.00 across runs; these descriptive differences do not establish a causal benefit from coordination.

An exploratory Jacobian-lens assay includes 150 receiver evaluations but no unauthorized adoption, leaving its intended participation-versus-refusal contrast unresolved. We release code and evaluation artifacts and identify assumptions auditors must check before treating dependence as evidence of collusion.

**Tracks to tick**
- Track 1: Markets and Collusion
- Track 2: Detection and Audit of Collusion

**Team name**
Rucha Agashe (solo)

**Location**
London, UK

**GitHub link**
https://github.com/in-a-quantum-world/collusion-net-influence

**Publishing question**
Yes

**Files to upload**
- Report PDF exported from the official template Google Doc (body 8 pages or fewer before references and appendices); project image: results/fig_extensions.png.
- Source: report/report.md, report/report_draft.docx, report/report_draft.pdf in the repository.

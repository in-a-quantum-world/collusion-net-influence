# Submission form fields (Apart Research AI Collusion Sprint, 26 October 2026)

**Project title**
When Net Influence Misleads: Calibration, Innocent Dependence, and Coalition Size

**Project summary** (249 words)

Pairwise net influence (Bonjour et al. 2022) detects a colluding pair from action logs alone: dependence between two agents' actions conditional on the public state, minus the strongest third-party dependence. We ask under which assumptions about public information, sample size and coalition structure it distinguishes planted coordination from innocent dependence, and where calibration fails to transfer. We extend the original ground-truth evaluation to matched innocent-dependence confounds, short-horizon calibration, state misspecification and a three-member coalition, over 7,400 scripted games, then test transfer on 30 games of GPT-4.1-mini agents with and without a hidden channel. Four failure mechanisms emerge. First, conditioning on the wrong public state confidently identifies an innocent pair that follows a shared public rule, in every game. Second, reducing the plug-in bias of the statistic is not the same as controlling false positives: the fixed threshold fires in half of all-random 25-round games, and a Holm-corrected permutation rule that is calibrated on the scripted nulls flags an innocent same-model pair on the pooled LLM control. Third, with a primary and two assistants, the pairwise net correction subtracts one coalition member's influence from another's and cancels the signal in the population limit, so more data makes the fixed-threshold detector less likely to fire. Fourth, on the LLM pilot, ranking, detection and benefit come apart: the team ranks first in nine of ten games, is flagged in five, and gained nothing. An exploratory Jacobian-lens assay on gemma-4-12B-it is reported with its intended positive arm absent. Code and logs are released.

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

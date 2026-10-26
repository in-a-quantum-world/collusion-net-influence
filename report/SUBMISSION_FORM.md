# Submission form fields (Apart Research AI Collusion Sprint, 26 October 2026)

**Project title**
Correlation is not agreement, and the workspace shows the action: calibrated net influence and a Jacobian-lens control for AI agent collusion

**Project summary** (246 words)

Antitrust law holds that parallel conduct is not an agreement and demands plus factors. We test one plus factor at two evidence surfaces, action logs and model internals, and measure how the naive reading of each mistakes something innocent for collusion. On actions, we implement pairwise net influence (Bonjour et al. 2022) for N players, add a within-state permutation null, and run the ground-truth evaluation the method has never had: 7,400 scripted games over five horizons, six collusion strengths, four baselines, two public-state definitions, two kinds of innocent pair and a coalition of three. Every unconditioned statistic ranks the innocent pair first in 96 to 100 percent of games; conditioning on the public state recovers the true pair in 98 to 100 percent; the permutation null cuts the 25-round false-positive rate from 100 to 25 percent; and we report where the detector breaks. On internals, we run the privileged-observer control Rose et al. (2026) proposed: a Jacobian-lens assay on gemma-4-12B-it with a partner's coded proposal and five matched conditions. The receiver never approved an ineligible item; the proposal reads out at rank 0 when the receiver is about to act on it legitimately and at chance when it has understood it and declined. A workspace monitor at the decision point sees the action, not the understanding. On GPT-4.1-mini agents with a hidden channel, the detector ranks the colluding pair first from actions alone and flags nothing in a same-model control. Code, logs and notebooks are released.

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
- Report PDF exported from the official template Google Doc (body 8 pages or fewer before references and appendices); project image: results/fig_evidence_ladder.png.
- Source: report/report.md, report/report_draft.docx, report/report_draft.pdf in the repository.

# Submission form fields (Apart Research AI Collusion Sprint, 26 October 2026)

**Project title**
Calibrated net influence: detecting a colluding pair from actions alone, and telling it apart from innocent parallel conduct

**Project summary** (248 words)

What should count as evidence that two AI agents are colluding? Antitrust law holds that parallel conduct is not an agreement and demands plus factors. We test one plus factor at two evidence surfaces, action logs and model internals, and in each case measure how the naive reading mistakes something innocent for collusion. On actions, we implement pairwise net influence (Bonjour et al. 2022) for N players, add a within-state permutation null, and run the ground-truth evaluation the method has never had: 4,400 scripted four-player games over five horizons, six collusion strengths, four baselines and two public-state definitions, an innocent pair following the same public rule, a harder innocent pair keyed on a richer state, and a coalition of three. The published memoryless statistic and every unconditioned baseline rank the innocent pair first in 96 to 100 percent of games; conditioning on the public state recovers the true pair in 98 to 100 percent; the permutation null cuts the 25-round false-positive rate from 100 percent to 25 percent; and we report where the conditioned detector breaks. Two pilots on real models were built but returned no results before the deadline, so no claim is made for them: an action-only test of GPT-4.1-mini agents with a hidden channel, and the privileged-observer control Rose et al. (2026) proposed, a Jacobian-lens assay on gemma-4-12B-it asking whether a partner's coded proposal reads out differently from a receiver that adopts it and one that understands and refuses. Code, logs and notebooks are released.

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
- Report PDF exported from the official template Google Doc (body 8 pages or fewer before references and appendices); figure results/fig_scripted.png is the main results figure.
- Source: report/report.md, report/report_draft.docx, report/report_draft.pdf in the repository.

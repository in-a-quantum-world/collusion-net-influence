# Submission form fields (Apart Research AI Collusion Sprint, 26 October 2026)

**Project title**
When Net Influence Misleads: Calibration, Innocent Dependence, and Coalition Size

**Project summary** (274 words)

When does statistical dependence between two agents count as evidence that they are colluding? We stress-test pairwise net influence (Bonjour et al. 2022), a detector that works from action logs alone, on 7,400 scripted games with a planted colluding pair and on two runs of GPT-4.1-mini agents, each ten games in three conditions.

We vary the public state the detector conditions on, the length of the game, and the size of the coalition. Two innocent agents that follow the same public rule defeat every unconditioned dependence statistic. Conditioning removes the problem only when the chosen state captures the rule they follow. At 25 rounds a fixed threshold accuses a pair in about half of all-random games. A Holm-corrected permutation rule holds the false-positive rate down on the scripted nulls, but flags a pair in each pooled LLM control that has no planted coalition. A rejected null is not the same as evidence of collusion.

With one primary agent and two assistants we derive a cancellation: the pairwise correction subtracts one coalition member's influence from another's, so fixed-threshold detection gets worse as the log grows.

In the LLM pilots the planted pair ranks first in eight or nine of ten games but is flagged in five or seven. Its mean score advantage was -0.25 in one run and +1.00 in the other, which does not show that the coordination paid off.

An exploratory Jacobian-lens experiment ran 150 receiver evaluations but produced no unauthorised adoption, so its intended contrast between taking part and refusing is unresolved. We release the code and logs and list the assumptions an auditor has to check before treating dependence as evidence.

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

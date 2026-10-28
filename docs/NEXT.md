# Next steps (adopted 26 Oct 2026)

Adopted from an external review of the submission on 26 October 2026.

## Immediately after the sprint
- Upload run 2's `llm_games.jsonl` to `results/20261026-1106-llm-pilot-run2/`.
- Tie audit of per-game rankings (unique-top recovery or fractional credit).
- Per-game reanalysis with at least 999 permutations so the Holm rule is informative on single games.
- Re-score run 1 with `experiments/analyse_llm.py` if its raw log is recovered.

## One targeted validation (scripted, no LLM calls)
Does coalition cancellation remain a practical detection failure once the detector gets a fair, independently calibrated threshold?
- Development/calibration games: all-random agents; innocent agents following the simple shared public rule.
- Untouched evaluation games: fresh games from those two conditions; the richer innocent rule; a two-member coalition; the three-member coalition.
- Rules compared: fixed rule (exactly one pair >= 0.05); Holm rule; a named score-based rule D_g = max_{i<j} s_ij with a threshold chosen on development games for a target any-pair false-alarm rate, frozen before evaluation.
- Report per horizon and condition: any-pair FPR, correct detection, unique-top recovery with ties handled, coalition-membership recovery (not just one edge), uncertainty across independently generated games. Hundreds of fresh evaluation games where feasible.
- Decisive figure: three panels (false accusations on honest behaviours; detection of two-member coalitions; detection of three-member coalitions as rounds grow), fixed and calibrated procedures together.

## Paper hierarchy
Main: stress test separating calibration failures, state-specification failures and coalition-size failures. Supporting: LLM pilot as a transfer check. Exploratory appendix: lens assay with its absent positive arm. Add a short "What should an auditor do?" box.

## Follow-up study (funding proposal)
"From Suspicious Dependence to Causal Evidence of Unauthorized Coordination": can an auditor identify when a partner's message causes an unauthorized action, without accusing legitimate cooperation, and do internal readouts add anything beyond an equally informed text monitor? Design in brief: an approval, allocation or verification task with explicit authorization; three behaviourally verified case types (unauthorized coordination carried out, legitimate cooperation, proposal actually rejected); message interventions (original, substituted, removed, sham); separate endpoints (decision, coordinated action, violations, benefit or harm, damage to legitimate cooperation); behavioural, text and internal monitors compared at equal information access; causal patching only if the readouts reproduce. Compute requested in stages tied to the gates below, starting with reproducibility and statistical review and a small pilot on one instrumented open model.

| Week | Work | Gate |
|---|---|---|
| 1 | Repair analysis, independent code review, calibrated scripted evaluation | Published results reproduce |
| 2 | New task (approval/allocation/verification with explicit authorization) and verified positive and negative controls; freeze protocol | Actual participation, legitimate cooperation and refusal all occur |
| 3-4 | Randomized message interventions on two model families | Effects reproduce on independent cases |
| 5 | Prespecified internal readouts; causal patches only if justified | Internals add value beyond text and survive controls |
| 6 | Held-out replication, uncertainty analysis, artifacts, report | Claims hold without selecting favourable examples |

# Source map and verification boundaries

The original study code and text-free inputs were published in this repository at commit `3b87b9dd0c55eb3ba6973bff9c8f9657364c60fb`. This revision adds analysis and audit material under `results/article/analysis-20260929/` without duplicating the inputs.

| Source | What it supports |
| --- | --- |
| `results/article/verdict-outcomes.csv` and `manifest.json` | Exact pseudonymous claim/pass predictions, validity, correctness, row count, and SHA-256. |
| `results/article/workflow-decisions.csv` and `manifest.json` | The 45,980 original workflow decisions and checksum. |
| `results/article/README.md` | Pass and cohort mappings. |
| `experiments/orchestration/averitec-controller-reliability-2026/config.yaml` | Tool stages, call budget, retry rule, and fault scenarios. |
| The study's `engine.py` | Expected-action oracle, counters, pre-dispatch fault behavior, and status handling. |
| The study's `extensions/evidence_selection/selector_instructions.txt` and adapters | Exact inclusion policy and interface wiring. |
| `analysis-manifest.json` | New output hashes and row counts. |

The source verdict CSV has SHA-256 `e2b2d3fe46090efd777e100df08e9bc23bfdc961bb2f7a42bcba9197a0d1ad83` and 2,700 rows. The original workflow decision CSV has SHA-256 `f3490a2ef830648ca1df32d528652f8f3f058186b87410a46e0bd76b60e94e7a` and 45,980 rows. The analysis verifies these inputs against the original manifest before recalculation. It does not verify how original model completions were generated.

The complete verdict analysis, numerical self-tests, every rounded Table 5 regression check, and four saved CSV comparisons were executed. Figure 2(c) was generated from those results. The workflow audit checks 849 nominal, 949 single-timeout, and 501 persistent-timeout checkpoints per complete run. The frozen-state audit compared expected actions and observation hashes across all 20 complete runs and found no differences. The reference observations yield 203 timeout-triggered and 96 invalid-result retries, plus 196 aborts after second failure (100 timeout and 96 invalid), and no budget-exhaustion aborts. The text-free exports make this accounting publicly checkable with `verify_workflow.py`; `workflow-observation-audit.json` records the private-ledger audit summary.

No models were rerun. No new relevance annotations, count-matched random controls, token or latency measurements, or live-trajectory comparisons were produced. The audit does not distinguish individual injected timeouts from other tool timeouts. Claim text, snippets, completions, and raw observations are not redistributed. The protocol for count-matched controls is future work and has no result in Tables 2–5.

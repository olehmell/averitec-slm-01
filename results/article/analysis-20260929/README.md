# Revised article analysis (2026-09-29)

This directory is the single public source for the CSIT revision's added analysis. It uses the pseudonymous records in the parent `results/article/` directory; there is no second copy of those inputs or separately maintained supplement. The original study files remain at their published paths. `analysis-manifest.json` gives SHA-256 checksums and row counts for these new outputs.

From the repository root, with Python 3.12+ and `uv`:

```bash
uv run --extra article python results/article/analysis-20260929/verify_revision.py
uv run python results/article/analysis-20260929/verify_workflow.py
uv run python results/article/analysis-20260929/workflow_audit.py --self-test
uv run python results/article/analysis-20260929/workflow_audit.py results/article/workflow-decisions.csv --manifest results/article/manifest.json --out /tmp/averitec-workflow-audit
```

The first command checks the input checksum and 2,700 rows, numerical self-tests, every rounded Table 5 value, and four saved verdict-analysis CSVs against fresh calculations. The second checks the original 45,980 workflow decisions and published text-free observation exports, confirms identical expected actions and observation hashes in all 20 complete runs, and recomputes 299 retries (203 after timeout, 96 after invalid), 1,700 other tool dispatches, and all 196 second-failure aborts per run. It requires no private logs. The last command regenerates the scenario audit. None of these commands reruns model inference.

`workflow-observations.csv` contains one pseudonymous reference checkpoint per case, scenario, and step. `workflow-state-hashes.csv` gives the matching expected action and SHA-256 observation digest for each of the 20 complete runs. These files disclose no claim, passage, completion, or tool-response text. `export_workflow_observations.py` and `frozen_workflow_audit.py` document how the files were checked against the private original run ledgers; re-exporting them requires those ledgers and the private workspace. The public `verify_workflow.py` checks the committed exports and their consistency with the original decision file, but cannot independently recover the private text from a hash.

`verdict-analysis/` contains the complete revised outputs, including per-class metrics and confusion matrices. `expected-table5.json` is the regression target for the manuscript table. `count-matched-protocol.md` describes a proposed, unrun control. `figures/` and `make_figure2.py` contain the revised figure and its construction code. `source-map.md` records provenance and verification limits.

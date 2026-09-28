# Final article results

From the repository root, run `uv run python reproduce_tables.py --check` to
recalculate CSIT Tables 2–5. The script checks file hashes and row counts
against [`manifest.json`](manifest.json), then compares the calculated values
with [`expected-tables.json`](expected-tables.json).

| File | Rows | Final measurement |
| --- | ---: | --- |
| [`reference.csv`](reference.csv) | 1,000 | Final candidate relevance grade (0, 1, or 2). |
| [`selection-decisions.csv`](selection-decisions.csv) | 21,000 | Seven selectors × three passes × 1,000 candidates. |
| [`verdict-outcomes.csv`](verdict-outcomes.csv) | 2,700 | Nine evidence sets × three passes × 100 cases, with validity and correctness. |
| [`workflow-decisions.csv`](workflow-decisions.csv) | 45,980 | Actions at the checkpoints included in Table 2. |

The final reference contains 233 direct-evidence, 516 contextual, and 251
irrelevant candidates. Jev's selector and HerO comparison passes 1, 2, and 3
use raw passes **1, 2, and 4**; `raw_pass` records this mapping. Table 2 uses
two complete Jev workflow measurements and three for each other deployment.
Gemini's three measurements are the historical run and extension passes 1 and
3; the `cohort` column identifies them.

Table 3 reports mean macro-F1 across three passes with 95% source-group
bootstrap intervals (2,000 resamples; seed 20260923). The IDs are stable
pseudonyms within this package. Claims, snippets, and model completions are
not included, so the package supports recalculation of the tables without
rerunning inference or inspecting source text.

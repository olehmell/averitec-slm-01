# AVeriTeC SLM 01

This repository contains the code, prompts, and results for the CSIT study of
workflow control, evidence selection, and HerO verdicts on 100 AVeriTeC-derived
cases. The seven deployments are Jev, Qwen-4B, Gemini, LFM-1.2B, LFM-2.6B,
Jeff, and Laya typed.

## Reproduce the article tables

With Python 3.12+ and [uv](https://docs.astral.sh/uv/), run:

```bash
uv run python reproduce_tables.py --check
```

The command recalculates Tables 2–5 from the released
[case- and candidate-level results](results/article/README.md). It requires no
model access or AVeriTeC source text.

## Files

- [Article results](results/article/README.md): final reference grades, selector
  decisions, workflow actions, verdict outcomes, and file hashes.
- [Workflow summary](experiments/orchestration/averitec-controller-reliability-2026/results/final/workflow.csv)
  and [HerO summary](experiments/orchestration/averitec-controller-reliability-2026/results/final/hero.csv):
  the run totals reported in the article.
- [Experiment code](experiments/orchestration/averitec-controller-reliability-2026/):
  controller, selector, and verifier implementations used in the study.

The released results contain pseudonymous IDs and measured outcomes. Claims,
source snippets, model completions, and credentials are not included. A new
model run requires the upstream AVeriTeC data, model access, and a new frozen
tool set; the released measurements are sufficient to verify the article's
numerical tables.

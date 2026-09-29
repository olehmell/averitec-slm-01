#!/usr/bin/env python3
"""Recompute verdict metrics and paired claim-cluster bootstrap intervals.

Run from the study repository, with NumPy installed:
  python complete_verdict_analysis.py results/article/verdict-outcomes.csv \
    --manifest results/article/manifest.json --out verdict-analysis

All passes of a claim stay together. Invalid predictions count against their
true class, without creating a fifth substantive verdict class. Intervals are
pointwise and exploratory, not multiplicity-adjusted. No inference is run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

LABELS = ('Supported', 'Refuted', 'Not Enough Evidence',
          'Conflicting Evidence/Cherrypicking')
EXPECTED = {'include_all', 'include_none', 'jev', 'gemini', 'qwen',
            'lfm26', 'jeff', 'laya_typed', 'lfm'}
NAMES = {'include_all': 'All snippets', 'include_none': 'No snippets',
         'jev': 'Jev', 'gemini': 'Gemini', 'qwen': 'Qwen-4B',
         'lfm26': 'LFM-2.6B', 'jeff': 'Jeff', 'laya_typed': 'Laya typed',
         'lfm': 'LFM-1.2B', 'majority_class': 'Always Refuted (reference policy)'}
ORDER = ('include_all', 'include_none', 'majority_class', 'jev', 'gemini',
         'qwen', 'lfm26', 'jeff', 'laya_typed', 'lfm')


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def f1_four(gold: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Last dimension is claims; returns macro-F1 for each preceding dimension."""
    total = np.zeros(pred.shape[:-1], dtype=float)
    for c in range(4):
        tp = np.sum((gold == c) & (pred == c), axis=-1)
        denominator = np.sum(gold == c, axis=-1) + np.sum(pred == c, axis=-1)
        total += np.divide(2.0 * tp, denominator, out=np.zeros_like(tp, dtype=float),
                           where=denominator != 0)
    return total / 4.0


def interval(values: np.ndarray, scale: float = 1.0) -> tuple[float, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low * scale), float(high * scale)


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)


def load(path: Path, manifest: Path | None, allow_subset: bool):
    if manifest:
        expected = json.loads(manifest.read_text())['files'][path.name]
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected['sha256'],
                'Input SHA-256 differs from manifest')
    with path.open(encoding='utf-8', newline='') as f:
        records = list(csv.DictReader(f))
    if manifest:
        require(len(records) == expected['rows'], 'Row count differs from manifest')
    require(bool(records), 'Empty input')
    arms = {r['evidence_set'] for r in records}
    require(arms <= EXPECTED, f'Unexpected evidence sets: {arms - EXPECTED}')
    require('include_all' in arms, 'The paired reference include_all is missing')
    require(allow_subset or arms == EXPECTED, 'Incomplete evidence sets; use --allow-subset only for an explicit partial analysis')
    index: dict[tuple[str, int, str], int] = {}
    gold_by_case: dict[str, int] = {}
    for r in records:
        arm, p, case = r['evidence_set'], int(r['comparison_pass']), r['case_id']
        require(p in (1, 2, 3), f'Unexpected pass {p}')
        require(int(r['raw_pass']) == (4 if arm == 'jev' and p == 3 else p), 'Unexpected raw-pass mapping')
        require(r['gold_label'] in LABELS, 'Unknown gold label')
        g = LABELS.index(r['gold_label'])
        require(case not in gold_by_case or gold_by_case[case] == g, 'Gold changes across conditions or passes')
        gold_by_case[case] = g
        require(r['valid'] in ('0', '1') and r['correct'] in ('0', '1'), 'Bad validity/correctness flag')
        valid = r['predicted_label'] in LABELS
        require(valid == (r['valid'] == '1'), 'Validity disagrees with predicted label')
        pred = LABELS.index(r['predicted_label']) if valid else -1
        require(int(r['correct']) == int(valid and pred == g), 'Correctness flag mismatch')
        key = (arm, p, case)
        require(key not in index, f'Duplicate record {key}')
        index[key] = pred
    cases = sorted(gold_by_case)
    require(len(cases) == 100, f'Expected 100 distinct claims, got {len(cases)}')
    require(len(index) == len(arms) * 3 * len(cases), 'Incomplete claim/pass grid')
    gold = np.array([gold_by_case[c] for c in cases], dtype=np.int8)
    predictions = {a: np.array([[index[(a, p, c)] for c in cases] for p in (1, 2, 3)], dtype=np.int8)
                   for a in arms}
    majority = int(np.bincount(gold, minlength=4).argmax())
    predictions['majority_class'] = np.full((3, len(cases)), majority, dtype=np.int8)
    # Three identical rows align the fixed reference policy with the estimand;
    # they are not inference runs or independent observations.
    return cases, gold, predictions, LABELS[majority]


def analyze(path: Path, out: Path, manifest: Path | None, b: int, seed: int,
            allow_subset: bool) -> None:
    cases, gold, predictions, majority_label = load(path, manifest, allow_subset)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(cases), size=(b, len(cases)))
    boot_gold = gold[sampled]
    points, boot = {}, {}
    for arm, pred in predictions.items():
        points[arm] = (float((pred == gold).mean()), float(f1_four(gold, pred).mean()))
        boot_accuracy = np.zeros(b)
        boot_f1 = np.zeros(b)
        for p in range(3):
            bp = pred[p][sampled]
            boot_accuracy += (bp == boot_gold).mean(axis=1) / 3
            boot_f1 += f1_four(boot_gold, bp) / 3
        boot[arm] = (boot_accuracy, boot_f1)
    summary, per_class, confusions, changes = [], [], [], []
    reference = predictions['include_all']
    for arm in ORDER:
        if arm not in predictions:
            continue
        pred = predictions[arm]
        acc, f1 = points[arm]
        da, df = acc - points['include_all'][0], f1 - points['include_all'][1]
        al, ah = interval(boot[arm][0], 100)
        fl, fh = interval(boot[arm][1])
        dl, dh = interval(boot[arm][0] - boot['include_all'][0], 100)
        dfl, dfh = interval(boot[arm][1] - boot['include_all'][1])
        summary.append(dict(evidence_set=arm, name=NAMES[arm],
            accuracy_pct=100*acc, accuracy_ci_low=al, accuracy_ci_high=ah,
            macro_f1=f1, macro_f1_ci_low=fl, macro_f1_ci_high=fh,
            delta_accuracy_vs_all_pp=100*da, delta_accuracy_ci_low=dl, delta_accuracy_ci_high=dh,
            delta_macro_f1_vs_all=df, delta_f1_ci_low=dfl, delta_f1_ci_high=dfh,
            mean_valid_per_100=float((pred >= 0).sum(axis=1).mean())))
        changes.append(dict(evidence_set=arm,
            mean_cases_corrected_vs_all=float(((pred == gold) & (reference != gold)).sum(axis=1).mean()),
            mean_cases_harmed_vs_all=float(((pred != gold) & (reference == gold)).sum(axis=1).mean()),
            mean_predictions_changed_vs_all=float((pred != reference).sum(axis=1).mean()),
            mean_invalid_per_100=float((pred < 0).sum(axis=1).mean())))
        for c, label in enumerate(LABELS):
            tp = np.sum((pred == c) & (gold == c), axis=1)
            n_pred = np.sum(pred == c, axis=1)
            n_gold = int(np.sum(gold == c))
            precision = np.divide(tp, n_pred, out=np.zeros(3, dtype=float), where=n_pred != 0)
            recall = tp / n_gold if n_gold else np.zeros(3)
            denom = n_pred + n_gold
            f1c = np.divide(2*tp, denom, out=np.zeros(3, dtype=float), where=denom != 0)
            per_class.append(dict(evidence_set=arm, label=label, support_per_pass=n_gold,
                mean_precision=float(precision.mean()), mean_recall=float(recall.mean()),
                mean_f1=float(f1c.mean())))
            for j, pred_label in [*enumerate(LABELS), (-1, 'Invalid/missing')]:
                confusions.append(dict(evidence_set=arm, gold_label=label, predicted_label=pred_label,
                    mean_cases_per_pass=float(((gold == c) & (pred == j)).sum(axis=1).mean())))
    write_csv(out/'verdict-summary.csv', summary)
    write_csv(out/'per-class-metrics.csv', per_class)
    write_csv(out/'confusion-matrices.csv', confusions)
    write_csv(out/'paired-changes.csv', changes)
    metadata = dict(input_file=str(path), input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    input_manifest_verified=bool(manifest), claims=len(cases), passes=3,
                    resampling_unit='claim; all conditions and three passes retained together',
                    bootstrap_replicates=b, seed=seed, numpy_version=np.__version__,
                    interval_type='pointwise 95% percentile; not multiplicity-adjusted',
                    labels=list(LABELS), gold_counts={l:int(sum(gold==i)) for i,l in enumerate(LABELS)},
                    majority_label=majority_label, majority_policy_is_inference=False,
                    partial_input=allow_subset)
    (out/'analysis-metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    lines = ['# Verdict analysis', '',
        'Mean of three pass-level scores. Each bootstrap sample retains all passes and conditions of a claim.',
        'Pointwise intervals are exploratory; repeating claims does not create 300 independent cases.', '',
        '| Condition | Accuracy, % (95% CI) | Macro-F1 (95% CI) | Δ accuracy vs all, pp (95% CI) |',
        '|---|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['name']} | {r['accuracy_pct']:.1f} ({r['accuracy_ci_low']:.1f}–{r['accuracy_ci_high']:.1f}) | "
                     f"{r['macro_f1']:.3f} ({r['macro_f1_ci_low']:.3f}–{r['macro_f1_ci_high']:.3f}) | "
                     f"{r['delta_accuracy_vs_all_pp']:+.1f} ({r['delta_accuracy_ci_low']:+.1f}–{r['delta_accuracy_ci_high']:+.1f}) |")
    lines += ['', f'Gold counts: {metadata["gold_counts"]}.',
              'The majority-class row is a fixed, sample-prevalence reference policy, not a HerO inference condition.',
              'A confidence interval crossing zero is not evidence of equivalence or non-inferiority.']
    (out/'verdict-summary.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


def self_test() -> None:
    gold = np.array([0, 1, 2, 3])
    require(float(f1_four(gold, gold)) == 1.0, 'Perfect prediction test')
    require(float(f1_four(gold, np.full(4, -1))) == 0.0, 'Invalid prediction test')
    require(abs(float(f1_four(gold, np.array([0, 1, 2, -1]))) - 0.75) < 1e-12,
            'Invalid prediction must penalize the true class, without adding a fifth class')
    require(abs(float(f1_four(gold, np.zeros(4, dtype=int))) - 0.1) < 1e-12, 'Constant prediction test')
    print('Self-tests passed.')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', nargs='?', type=Path)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--out', type=Path, default=Path('verdict-analysis'))
    p.add_argument('--bootstrap', type=int, default=10000)
    p.add_argument('--seed', type=int, default=20260929)
    p.add_argument('--allow-subset', action='store_true')
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    if args.self_test:
        self_test()
    if args.input:
        require(args.bootstrap >= 100, 'Use at least 100 bootstrap replicates')
        analyze(args.input, args.out, args.manifest, args.bootstrap, args.seed, args.allow_subset)
    elif not args.self_test:
        p.error('Provide an input CSV or --self-test')

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Recompute the complete analysis and check every numeric Table 5 entry.

Run from any directory: python /path/to/analysis-20260929/verify_revision.py
The test reads the pinned input, recalculates all intervals, checks the original
manuscript's rounded numbers and compares the bundled output CSVs byte for byte.
"""
from pathlib import Path
import csv, json, subprocess, sys, tempfile

ROOT=Path(__file__).resolve().parent
ARTICLE=ROOT.parent
FIELDS=[('accuracy',1,['accuracy_pct','accuracy_ci_low','accuracy_ci_high']),
        ('delta',1,['delta_accuracy_vs_all_pp','delta_accuracy_ci_low','delta_accuracy_ci_high']),
        ('macro_f1',3,['macro_f1','macro_f1_ci_low','macro_f1_ci_high'])]

def main():
    expected=json.loads((ROOT/'expected-table5.json').read_text())
    records=list(csv.DictReader((ARTICLE/'verdict-outcomes.csv').open()))
    counts={}
    for r in records:
        key=(r['evidence_set'],int(r['comparison_pass']))
        counts[key]=counts.get(key,0)+int(r['correct'])
    with tempfile.TemporaryDirectory() as folder:
        out=Path(folder)
        subprocess.run([sys.executable,str(ROOT/'complete_verdict_analysis.py'),str(ARTICLE/'verdict-outcomes.csv'),
                        '--manifest',str(ARTICLE/'manifest.json'),'--out',str(out),'--self-test'],check=True,stdout=subprocess.DEVNULL)
        result={r['evidence_set']:r for r in csv.DictReader((out/'verdict-summary.csv').open())}
        for arm,e in expected.items():
            r=result[arm]
            for name,digits,fields in FIELDS:
                got=[round(float(r[k]),digits) for k in fields]
                if got!=e[name]:raise ValueError(f'Table 5 mismatch: {arm}/{name}: {got} != {e[name]}')
            if arm!='majority_class':
                got=[counts[(arm,p)] for p in (1,2,3)]
                if got!=e['correct']:raise ValueError(f'Pass count mismatch: {arm}')
        for name in ('verdict-summary.csv','per-class-metrics.csv','confusion-matrices.csv','paired-changes.csv'):
            if (out/name).read_bytes()!=(ROOT/'verdict-analysis'/name).read_bytes():
                raise ValueError(f'Bundled output differs from fresh calculation: {name}')
    print('PASS: input manifest, 2,700 records, self-tests, every Table 5 value and all four analysis CSVs.')
    print('Scope: verdict reanalysis only. Run verify_workflow.py for public workflow accounting.')

if __name__=='__main__':main()

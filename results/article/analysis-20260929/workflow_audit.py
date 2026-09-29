#!/usr/bin/env python3
"""Audit scenario counts without inferring unrecorded reasons for termination.

python workflow_audit.py /repo/results/article/workflow-decisions.csv \
    --manifest /repo/results/article/manifest.json --out workflow-audit

This utility has synthetic self-tests and was executed on the full source
workflow decision CSV in the final manuscript revision. Per-trial observation
traces are needed to attribute aborts to budget or second failure and injected
versus other faults.
"""
from __future__ import annotations
import argparse,csv,hashlib,json
from collections import defaultdict,Counter
from pathlib import Path

STAGES={'decompose','queries','retrieve','qa','coverage','select','verdict'}
REQUIRED={'deployment','measurement','case_id','scenario','step','expected_action','selected_action','provider_outcome','compliant'}

def trial_counts(rows):
    rows=sorted(rows,key=lambda r:int(r['step']))
    steps=[int(r['step']) for r in rows]
    if not rows or steps!=list(range(steps[0],steps[0]+len(steps))):raise ValueError('Non-contiguous or duplicate trial steps')
    expected=[r['expected_action'] for r in rows]
    if any(a not in STAGES|{'abort','finish'} for a in expected):raise ValueError('Unknown reference action')
    if expected[-1] not in {'abort','finish'} or any(a in {'abort','finish'} for a in expected[:-1]):
        raise ValueError('Trial must have exactly one terminal reference action, at its end')
    retry=sum(a in STAGES and a==b for a,b in zip(expected[1:],expected[:-1]))
    # Repeated reference tool dispatches are observable. Why a tool failed is not.
    dispatches=sum(a in STAGES for a in expected)
    return Counter(trials=1,checkpoints=len(rows),tool_dispatches=dispatches,
                   repeated_reference_tool=retry,other_tool_dispatches=dispatches-retry,
                   abort=expected.count('abort'),finish=expected.count('finish'))

def run(path,manifest,out):
    payload=path.read_bytes()
    if manifest:
        wanted=json.loads(manifest.read_text())['files'][path.name]
        if hashlib.sha256(payload).hexdigest()!=wanted['sha256']:raise ValueError('Source hash mismatch')
    with path.open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        if not REQUIRED<=set(reader.fieldnames or []):raise ValueError(f'Missing columns: {REQUIRED-set(reader.fieldnames or [])}')
        rows=list(reader)
    if manifest and len(rows)!=wanted['rows']:raise ValueError('Source row count mismatch')
    by_trial=defaultdict(list)
    for r in rows:
        if r['compliant'] not in ('0','1'):raise ValueError('Invalid compliance value')
        if (r['compliant']=='1')!=(r['provider_outcome']=='ok' and r['selected_action']==r['expected_action']):raise ValueError('Compliance flag mismatch')
        by_trial[(r['deployment'],int(r['measurement']),r['case_id'],r['scenario'])].append(r)
    summary=defaultdict(Counter)
    for (arm,runno,case,scenario),events in by_trial.items():summary[(arm,runno,scenario)].update(trial_counts(events))
    report=[];totals=defaultdict(Counter)
    for (arm,runno,scenario),c in sorted(summary.items()):
        report.append({'deployment':arm,'measurement':runno,'scenario':scenario,**dict(c),
                       'abort_cause_status':'not attributed; inspect frozen observations'})
        totals[(arm,runno)].update(c)
    out.mkdir(parents=True,exist_ok=True)
    with (out/'scenario-counts.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(report[0]));w.writeheader();w.writerows(report)
    checks=[]
    for key,c in sorted(totals.items()):
        match=all(c[k]==v for k,v in {'trials':300,'checkpoints':2299,'tool_dispatches':1999,
                                     'abort':196,'finish':104,'repeated_reference_tool':299,
                                     'other_tool_dispatches':1700}.items())
        checks.append({'deployment':key[0],'measurement':key[1],**dict(c),'matches_reported_accounting':match})
    meta={'input_sha256':hashlib.sha256(payload).hexdigest(),'rows':len(rows),'run_checks':checks,
          'abort_causes':'not inferred from scenario labels; observation traces required',
          'retry_definition':'299 consecutive identical nonterminal expected actions; 1,700 other tool dispatches',
          'timeout_invalid_split':'see workflow-observation-audit.json for the verified 203/96 split across 20 identical frozen-state runs'}
    (out/'audit.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(checks,indent=2))
    if not all(x['matches_reported_accounting'] for x in checks):raise ValueError('Reported accounting differs; inspect audit.json before amending the manuscript')

def self_test():
    def rows(actions):return [{'step':str(i),'expected_action':a} for i,a in enumerate(actions)]
    c=trial_counts(rows(['decompose','queries','retrieve','retrieve','abort']))
    assert c['checkpoints']==5 and c['repeated_reference_tool']==1 and c['other_tool_dispatches']==3 and c['abort']==1
    c=trial_counts(rows(['decompose','queries','retrieve','qa','coverage','select','verdict','finish']))
    assert c['checkpoints']==8 and c['finish']==1 and c['repeated_reference_tool']==0
    try:trial_counts(rows(['decompose','finish','retrieve']))
    except ValueError:pass
    else:raise AssertionError('Early terminal test failed')
    print('Workflow audit synthetic tests passed. Full source workflow audit not implied.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path,nargs='?');p.add_argument('--manifest',type=Path);p.add_argument('--out',type=Path,default=Path('workflow-audit'));p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:self_test()
    if a.input:run(a.input,a.manifest,a.out)
    elif not a.self_test:p.error('Supply input CSV or --self-test')

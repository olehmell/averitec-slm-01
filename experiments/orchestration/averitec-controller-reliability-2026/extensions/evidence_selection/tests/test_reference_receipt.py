import importlib.util
import json
from pathlib import Path
import sys
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import annotation as a
import reference_receipt as r


def fixture(root):
    data = {'schema':'averitec-frozen-evidence-candidates/v1', 'gold_included':False,
            'source_freeze_sha256':'a'*64, 'source_traces_sha256':'b'*64,
            'cases':[{'case_id':f'case-{i}', 'group_id':f'group-{i}', 'claim':'Synthetic assertion',
                      'candidates':[{'id':f'C{j:02d}', 'passage_id':f'p{j}', 'text':'Synthetic passage', 'url':'synthetic',
                                     'source_start':0, 'source_text_length':17, 'source_text_sha256':'c'*64}
                                    for j in range(1,11)]} for i in range(100)]}
    (root/'candidates.json').write_text(r.canonical(data))
    (root/'annotation-inputs').mkdir()
    for batch in range(1,11):
        presented = [{'case_id':c['case_id'], 'claim':c['claim'],
                      'candidates':[{k:p[k] for k in ('id','text','url')} for p in c['candidates']]}
                     for c in data['cases'][(batch-1)*10:batch*10]]
        (root/f'annotation-inputs/batch-{batch:02d}.json').write_text(r.canonical(presented))
    (root/'adjudication-inputs').mkdir()
    for batch in range(1,7):
        (root/f'adjudication-inputs/batch-{batch:02d}.json').write_text('[]')
    for side in ('a','b'):
        target = root/'annotations'/side
        target.mkdir(parents=True)
        for batch in range(1,11):
            rows = [{'case_id':c['case_id'], 'ratings':[{'id':p['id'], 'grade':2, 'reason':'Synthetic fixture.'}
                                                      for p in c['candidates']]}
                    for c in data['cases'][(batch-1)*10:batch*10]]
            (target/f'batch-{batch:02d}.json').write_text(r.canonical(rows))
    for side, numbers in (('first',range(1,4)), ('last',range(4,7))):
        target = root/'adjudication'/side
        target.mkdir(parents=True)
        for batch in numbers:
            (target/f'batch-{batch:02d}.json').write_text('[]')
    expected = a.validate_candidates(data)
    rec = a.reconcile(expected, a.load_annotations(root/'annotations/a', expected),
                      a.load_annotations(root/'annotations/b', expected))
    rec['input_sha256'] = a.sha256(root/'candidates.json')
    rec['annotation_files_sha256'] = {str(p.relative_to(root/'annotations')):a.sha256(p)
                                    for p in sorted((root/'annotations').glob('*/batch-*.json'))}
    (root/'reconciliation.json').write_text(r.canonical(rec))
    (root/'tokenizer-preflight.json').write_text(r.canonical({
        'complete':True, 'candidates_sha256':a.sha256(root/'candidates.json'),
        'instructions_sha256':a.sha256(HERE/'selector_instructions.txt')}))


def test_reference_build_verify_and_tamper_detection(tmp_path):
    fixture(tmp_path)
    reference, receipt = r.build(tmp_path)
    (tmp_path/'reference.json').write_text(reference)
    (tmp_path/'reference-receipt.json').write_text(receipt)
    assert r.verify(tmp_path)['known_labels'] == 1000
    altered = json.loads(reference)
    altered['ratings'][0]['grade'] = 0
    (tmp_path/'reference.json').write_text(r.canonical(altered))
    with pytest.raises(ValueError, match='reference_content_mismatch'):
        r.verify(tmp_path)


def test_mutated_annotation_cannot_silently_reseal(tmp_path):
    fixture(tmp_path)
    path = tmp_path/'annotations/a/batch-01.json'
    rows = json.loads(path.read_text())
    rows[0]['ratings'][0]['grade'] = 0
    path.write_text(r.canonical(rows))
    with pytest.raises(ValueError, match='reconciliation_drift'):
        r.build(tmp_path)


def test_swapped_judge_input_is_rejected(tmp_path):
    fixture(tmp_path)
    path = tmp_path/'annotation-inputs/batch-01.json'
    rows = json.loads(path.read_text())
    rows[0]['candidates'][0]['text'] = 'Other text'
    path.write_text(r.canonical(rows))
    with pytest.raises(ValueError, match='annotation_input_presentation_drift'):
        r.build(tmp_path)

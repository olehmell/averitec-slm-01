import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('annotation', Path(__file__).resolve().parents[1] / 'annotation.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def fixtures():
    data = {'schema':'averitec-frozen-evidence-candidates/v1', 'gold_included':False,
            'source_freeze_sha256':'a'*64, 'source_traces_sha256':'b'*64,
            'cases':[{'case_id':f'case-{i}', 'group_id':f'group-{i}', 'claim':'A claim',
                      'candidates':[{'id':f'C{j:02d}', 'passage_id':f'p{j}', 'text':'text', 'url':'source',
                                     'source_start':0, 'source_text_length':4, 'source_text_sha256':'c'*64}
                                    for j in range(1,11)]} for i in range(100)]}
    expected = a.validate_candidates(data)
    ratings = {k:{'id':k[1], 'grade':2, 'reason':'Direct information.'} for k in expected}
    return data, expected, ratings


def test_integrity():
    data, expected, ratings = fixtures()
    assert len(expected) == 1000
    data['gold_included'] = True
    with pytest.raises(ValueError): a.validate_candidates(data)


@pytest.mark.parametrize('field', ['gold_label', 'rating', 'expected_action'])
@pytest.mark.parametrize('level', ['root', 'case', 'candidate'])
def test_extra_label_fields_rejected(field, level):
    data, _, _ = fixtures()
    target = {'root':data, 'case':data['cases'][0], 'candidate':data['cases'][0]['candidates'][0]}[level]
    target[field] = 2
    with pytest.raises(ValueError, match='candidate_schema_or_gold'):
        a.validate_candidates(data)


def test_disagreements_unknown_and_finalization():
    _, expected, first = fixtures()
    second = {k:dict(v) for k,v in first.items()}
    keys = list(expected)
    second[keys[0]]['grade'] = 1
    second[keys[1]]['grade'] = 'U'
    result = a.reconcile(expected, first, second)
    assert len(result['disputes']) == 2
    assert result['exact_grade_agreement_count'] == 998
    assert result['binary_agreement_denominator'] == 999
    with pytest.raises(ValueError): a.finalize(result, [])
    adjudication = [{'case_id':k[0], 'id':k[1], 'grade':'U', 'reason':'Insufficient context.'} for k in keys[:2]]
    final = a.finalize(result, adjudication)
    assert final['known_count'] == 998 and final['unknown_count'] == 2


def test_loader_rejects_bool_duplicate_unknown_and_missing(tmp_path):
    _, expected, _ = fixtures()
    rows = [{'case_id':'case-0', 'ratings':[{'id':f'C{j:02d}', 'grade':2, 'reason':'Direct.'} for j in range(1,11)]}]
    f = tmp_path / 'batch-01.json'
    f.write_text(json.dumps(rows))
    assert len(a.load_annotations(tmp_path, expected, complete=False)) == 10
    with pytest.raises(ValueError): a.load_annotations(tmp_path, expected)
    rows[0]['ratings'][0]['grade'] = True
    f.write_text(json.dumps(rows))
    with pytest.raises(ValueError): a.load_annotations(tmp_path, expected, complete=False)
    rows[0]['ratings'][0]['grade'] = 2
    f.write_text(json.dumps(rows + rows))
    with pytest.raises(ValueError): a.load_annotations(tmp_path, expected, complete=False)

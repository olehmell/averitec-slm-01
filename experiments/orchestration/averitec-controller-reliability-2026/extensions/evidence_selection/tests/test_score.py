import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('evidence_score', Path(__file__).resolve().parents[1] / 'score.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


def reference():
    return {'reference_type':'ai_annotated_not_human_gold', 'ratings':[
        {'case_id':'a', 'id':'C01', 'grade':2},
        {'case_id':'a', 'id':'C02', 'grade':1},
        {'case_id':'b', 'id':'C01', 'grade':0},
        {'case_id':'b', 'id':'C02', 'grade':'U'}]}


def test_exact_predictions_and_unknown_labels():
    outputs = [dict(case_id=r['case_id'], id=r['id'], outcome='ok', action='include' if r['grade']==2 else 'exclude')
               for r in reference()['ratings']]
    out = s.score(reference(), outputs)
    assert out['micro']['f1'] == 1
    assert out['no_positive_case_empty_selection_accuracy'] == 1
    assert out['known_labels'] == 3 and out['unknown_labels'] == 1
    assert s.score(reference(), outputs, positive_grades=(1,2))['micro']['recall'] == .5


def test_missing_is_not_correct_exclusion():
    out = s.score(reference(), [])
    assert not out['headline_metrics_ready']
    assert out['micro'] is None
    assert out['correct_over_all_known_labels'] == 0
    assert out['missing_known_responses'] == 3


def test_error_with_action_is_not_success():
    outputs = [dict(case_id=r['case_id'], id=r['id'], outcome='transport_error', action='exclude')
               for r in reference()['ratings']]
    out = s.score(reference(), outputs)
    assert out['invalid_known_responses'] == 3
    assert out['valid_known_responses'] == 0
    assert out['correct_over_all_known_labels'] == 0
    with pytest.raises(ValueError): s.score(reference(), outputs+outputs)


def test_prediction_gold_field_rejected():
    row = {'case_id':'a', 'id':'C01', 'outcome':'ok', 'action':'include', 'gold':2}
    with pytest.raises(ValueError, match='prediction_schema_or_gold'):
        s.score(reference(), [row])

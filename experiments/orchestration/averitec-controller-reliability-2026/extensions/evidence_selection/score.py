"""Offline evidence-selection metrics; missing outputs never become true negatives."""
from collections import Counter, defaultdict


def rates(counts):
    tp, fp, fn = (counts.get(k, 0) for k in ('tp', 'fp', 'fn'))
    return {'precision': tp / (tp+fp) if tp+fp else None,
            'recall': tp / (tp+fn) if tp+fn else None,
            'f1': 2*tp / (2*tp+fp+fn) if 2*tp+fp+fn else None}


def score(reference, predictions, *, positive_grades=(2,)):
    """Prediction rows: case_id, id, action, outcome. No model calls or repair.

    Headline F1 requires a valid response for EVERY known label. Otherwise only
    explicitly conditional metrics are reported, alongside coverage and accuracy
    with all unknown/missing outcomes counted as incorrect.
    """
    labels = {}
    for row in reference['ratings']:
        key = (row['case_id'], row['id'])
        if key in labels:
            raise ValueError('duplicate_reference')
        labels[key] = row['grade']
    outputs = {}
    for row in predictions:
        if set(row) != {'case_id', 'id', 'action', 'outcome'}:
            raise ValueError('prediction_schema_or_gold')
        key = (row['case_id'], row['id'])
        if key not in labels or key in outputs:
            raise ValueError('unknown_or_duplicate_prediction')
        outputs[key] = row
    matrix = Counter()
    per_case = defaultdict(Counter)
    missing = invalid = uncertain = known = valid = correct = 0
    for key, grade in labels.items():
        if grade == 'U':
            uncertain += 1
            continue
        if type(grade) is not int or grade not in (0,1,2):
            raise ValueError('invalid_reference_grade')
        known += 1
        case = per_case[key[0]]
        case['known'] += 1
        truth = grade in positive_grades
        case['positives'] += truth
        row = outputs.get(key)
        if row is None:
            missing += 1
            case['failed'] += 1
            continue
        if row.get('outcome') != 'ok' or row.get('action') not in ('include', 'exclude'):
            invalid += 1
            case['failed'] += 1
            continue
        predicted = row['action'] == 'include'
        valid += 1
        correct += predicted == truth
        item = ('tp' if truth else 'fp') if predicted else ('fn' if truth else 'tn')
        matrix[item] += 1
        case[item] += 1
        case['selected'] += predicted
    primary_ready = known > 0 and missing == invalid == 0
    positive_cases = [v for v in per_case.values() if v['positives'] and not v['failed']]
    zero_cases = [v for v in per_case.values() if not v['positives'] and not v['failed']]
    return {
        'reference_type': reference['reference_type'],
        'positive_grades': list(positive_grades),
        'planned_candidates': len(labels), 'known_labels': known, 'unknown_labels': uncertain,
        'known_label_coverage': known / len(labels) if labels else 0,
        'valid_known_responses': valid, 'missing_known_responses': missing,
        'invalid_known_responses': invalid,
        'valid_known_response_coverage': valid / known if known else 0,
        'correct_over_all_known_labels': correct / known if known else None,
        'headline_metrics_ready': primary_ready,
        'micro': rates(matrix) if primary_ready else None,
        'conditional_on_valid_known_responses': {'counts': dict(matrix), **rates(matrix)},
        'case_macro_f1_positive_cases': (sum(rates(v)['f1'] for v in positive_cases) / len(positive_cases))
            if primary_ready and positive_cases else None,
        'case_macro_f1_positive_case_count': len(positive_cases),
        'no_positive_case_empty_selection_accuracy':
            sum(v['selected'] == 0 for v in zero_cases) / len(zero_cases)
            if primary_ready and zero_cases else None,
        'no_positive_case_count': len(zero_cases),
        'caveat': 'Pointwise selection against AI relevance labels, not factual-verdict accuracy or full-corpus recall.',
    }

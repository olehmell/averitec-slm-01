"""Strict, offline validation and reconciliation of two blind AI annotations."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_candidates(data: dict) -> dict[tuple[str, str], dict]:
    root_keys = {'schema', 'source_freeze_sha256', 'source_traces_sha256', 'gold_included', 'cases'}
    if (not isinstance(data, dict) or set(data) != root_keys
            or data.get('schema') != 'averitec-frozen-evidence-candidates/v1'
            or data.get('gold_included') is not False):
        raise ValueError('candidate_schema_or_gold')
    result = {}
    seen = set()
    for case in data['cases']:
        if not isinstance(case, dict) or set(case) != {'case_id', 'group_id', 'claim', 'candidates'}:
            raise ValueError('candidate_schema_or_gold')
        cid = case['case_id']
        if cid in seen or not isinstance(case['claim'], str) or not case['claim'].strip():
            raise ValueError('duplicate_case_or_empty_claim')
        seen.add(cid)
        if len(case['candidates']) != 10:
            raise ValueError('candidate_count')
        if len({p['passage_id'] for p in case['candidates']}) != 10:
            raise ValueError('duplicate_passage')
        for i, candidate in enumerate(case['candidates'], 1):
            if not isinstance(candidate, dict) or set(candidate) != {
                'id', 'passage_id', 'text', 'url', 'source_start', 'source_text_length', 'source_text_sha256'
            }:
                raise ValueError('candidate_schema_or_gold')
            if candidate['id'] != f'C{i:02d}' or not candidate['text'].strip():
                raise ValueError('candidate_id_or_empty_text')
            result[(cid, candidate['id'])] = {'claim': case['claim'], **candidate}
    if len(seen) != 100 or len(result) != 1000:
        raise ValueError('expected_100_cases_1000_candidates')
    return result


def load_annotations(directory: Path, expected: dict, *, complete: bool = True) -> dict:
    ratings = {}
    case_ids = set()
    for path in sorted(directory.glob('batch-*.json')):
        batch = json.loads(path.read_text())
        if not isinstance(batch, list):
            raise ValueError('annotation_batch_not_list')
        for case in batch:
            if set(case) != {'case_id', 'ratings'} or case['case_id'] in case_ids:
                raise ValueError('annotation_case_schema_or_duplicate')
            case_ids.add(case['case_id'])
            if len(case['ratings']) != 10:
                raise ValueError('annotation_case_rating_count')
            for rating in case['ratings']:
                if set(rating) != {'id', 'grade', 'reason'}:
                    raise ValueError('annotation_rating_schema')
                key = (case['case_id'], rating['id'])
                if key not in expected or key in ratings:
                    raise ValueError('annotation_unknown_or_duplicate_id')
                grade = rating['grade']
                if not ((type(grade) is int and grade in (0, 1, 2)) or grade == 'U'):
                    raise ValueError('annotation_grade')
                if not isinstance(rating['reason'], str) or not rating['reason'].strip():
                    raise ValueError('annotation_reason')
                ratings[key] = rating
    if complete and set(ratings) != set(expected):
        raise ValueError(f'incomplete_annotations:{len(ratings)}/{len(expected)}')
    return ratings


def reconcile(expected: dict, a: dict, b: dict) -> dict[str, Any]:
    if set(a) != set(expected) or set(b) != set(expected):
        raise ValueError('incomplete_paired_annotations')
    consensus, disputes = [], []
    matrix = Counter()
    exact = binary = unknown = 0
    for key, candidate in expected.items():
        ra, rb = a[key], b[key]
        ga, gb = ra['grade'], rb['grade']
        matrix[f'{ga}:{gb}'] += 1
        exact += ga == gb
        known = ga != 'U' and gb != 'U'
        unknown += not known
        binary += known and ((ga == 2) == (gb == 2))
        row = {'case_id': key[0], 'id': key[1]}
        if known and ga == gb:
            consensus.append({**row, 'grade': ga, 'decision_source': 'two_blind_passes_agree'})
        else:
            disputes.append({**row, 'claim': candidate['claim'], 'text': candidate['text'],
                             'url': candidate['url'], 'reviewer_a': ra, 'reviewer_b': rb})
    return {'schema': 'averitec-ai-annotation-reconciliation/v1',
            'reference_type': 'ai_annotated_not_human_gold',
            'paired_candidates': len(expected), 'exact_grade_agreement_count': exact,
            'binary_agreement_count_excluding_unknown': binary,
            'binary_agreement_denominator': len(expected) - unknown,
            'any_unknown_count': unknown, 'grade_confusion_counts': dict(sorted(matrix.items())),
            'consensus': consensus, 'disputes': disputes, 'final_reference_ready': not disputes}


def finalize(reconciliation: dict, adjudications: list[dict]) -> dict:
    disputes = {(r['case_id'], r['id']): r for r in reconciliation['disputes']}
    seen = set()
    final = list(reconciliation['consensus'])
    for row in adjudications:
        if set(row) != {'case_id', 'id', 'grade', 'reason'}:
            raise ValueError('adjudication_schema')
        key = (row['case_id'], row['id'])
        if key not in disputes or key in seen:
            raise ValueError('adjudication_unknown_or_duplicate')
        grade = row['grade']
        if not ((type(grade) is int and grade in (0, 1, 2)) or grade == 'U'):
            raise ValueError('adjudication_grade')
        if not isinstance(row['reason'], str) or not row['reason'].strip():
            raise ValueError('adjudication_reason')
        seen.add(key)
        final.append({**row, 'decision_source': 'adjudicated_after_blind_passes'})
    if seen != set(disputes):
        raise ValueError('incomplete_adjudication')
    return {'schema': 'averitec-ai-evidence-reference/v1',
            'reference_type': 'ai_annotated_not_human_gold',
            'candidate_count': len(final),
            'known_count': sum(r['grade'] != 'U' for r in final),
            'unknown_count': sum(r['grade'] == 'U' for r in final),
            'grade_counts': dict(Counter(str(r['grade']) for r in final)),
            'ratings': sorted(final, key=lambda r: (r['case_id'], r['id']))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--progress', action='store_true')
    args = parser.parse_args()
    expected = validate_candidates(json.loads(args.candidates.read_text()))
    ratings = {x: load_annotations(args.annotations / x, expected, complete=not args.progress)
               for x in ('a', 'b')}
    if args.progress:
        output = {'expected':len(expected), **{k:len(v) for k,v in ratings.items()}}
    else:
        output = reconcile(expected, ratings['a'], ratings['b'])
        output['input_sha256'] = sha256(args.candidates)
        output['annotation_files_sha256'] = {
            str(p.relative_to(args.annotations)): sha256(p)
            for p in sorted(args.annotations.glob('*/batch-*.json'))}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

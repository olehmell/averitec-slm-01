"""Create/verify a deterministic integrity receipt for the AI reference.

All commands are offline. Creation emits text to stdout; it does not overwrite
annotations or reference files. This is integrity verification, not a signature
or evidence that AI relevance labels are human ground truth.
"""
import argparse
import hashlib
import json
from pathlib import Path

from annotation import finalize, load_annotations, reconcile, sha256, validate_candidates


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n'


def reference_inputs(root):
    fixed = ['candidates.json', 'reconciliation.json', 'tokenizer-preflight.json']
    fixed += [f'annotation-inputs/batch-{n:02d}.json' for n in range(1,11)]
    fixed += [f'adjudication-inputs/batch-{n:02d}.json' for n in range(1,7)]
    fixed += [f'annotations/{side}/batch-{n:02d}.json' for side in ('a','b') for n in range(1,11)]
    fixed += [f'adjudication/first/batch-{n:02d}.json' for n in range(1,4)]
    fixed += [f'adjudication/last/batch-{n:02d}.json' for n in range(4,7)]
    if any(not (root / name).is_file() for name in fixed):
        raise ValueError('incomplete_reference_inputs')
    return {name:sha256(root / name) for name in fixed}


def build(root):
    files = reference_inputs(root)
    data = json.loads((root / 'candidates.json').read_text())
    expected = validate_candidates(data)
    for n in range(1,11):
        presented = [{'case_id':c['case_id'], 'claim':c['claim'],
                      'candidates':[{key:p[key] for key in ('id','text','url')} for p in c['candidates']]}
                     for c in data['cases'][(n-1)*10:n*10]]
        if json.loads((root/f'annotation-inputs/batch-{n:02d}.json').read_text()) != presented:
            raise ValueError('annotation_input_presentation_drift')
    a = load_annotations(root/'annotations/a', expected)
    b = load_annotations(root/'annotations/b', expected)
    fresh = reconcile(expected, a, b)
    recorded = json.loads((root/'reconciliation.json').read_text())
    for key, value in fresh.items():
        if recorded.get(key) != value:
            raise ValueError('reconciliation_drift')
    for n in range(1,7):
        if json.loads((root/f'adjudication-inputs/batch-{n:02d}.json').read_text()) != fresh['disputes'][(n-1)*50:n*50]:
            raise ValueError('adjudication_input_presentation_drift')
    if recorded.get('input_sha256') != files['candidates.json']:
        raise ValueError('reconciliation_input_binding')
    annotation_hashes = {name.removeprefix('annotations/'):digest
                         for name,digest in files.items() if name.startswith('annotations/')}
    if recorded.get('annotation_files_sha256') != annotation_hashes:
        raise ValueError('reconciliation_annotation_binding')
    adjudications = []
    for name in files:
        if name.startswith('adjudication/'):
            rows = json.loads((root/name).read_text())
            if not isinstance(rows, list):
                raise ValueError('adjudication_not_list')
            adjudications.extend(rows)
    reference = finalize(fresh, adjudications)
    if reference['candidate_count'] != 1000:
        raise ValueError('final_reference_count')
    reference['source_candidates_sha256'] = files['candidates.json']
    reference['annotation_method'] = {
        'initial_passes':'two_separate_sessions_blind_to_peer_ratings_and_selector_predictions',
        'requested_models':['gpt-5.6-sol','gpt-6-astra'],
        'adjudication_requested_model':'gpt-6-astra',
        'adjudication_scope':'all_grade_disagreements_and_any_U',
        'shared_provider_bias':'not_independent_model_families',
        'human_validation':False,
        'external_sources_used':False,
        'relevance_not_verified_source_truth':True,
    }
    preflight = json.loads((root/'tokenizer-preflight.json').read_text())
    if preflight.get('candidates_sha256') != files['candidates.json'] or preflight.get('complete') is not True:
        raise ValueError('token_preflight_candidate_binding')
    here = Path(__file__).resolve().parent
    code_hashes = {name:sha256(here/name) for name in (
        'annotation.py', 'reference_receipt.py', 'config.yaml', 'selector_instructions.txt')}
    if preflight.get('instructions_sha256') != code_hashes['selector_instructions.txt']:
        raise ValueError('token_preflight_instructions_binding')
    text = canonical(reference)
    receipt = {
        'schema':'averitec-ai-reference-integrity/v1',
        'complete':True,
        'reference_type':'ai_annotated_not_human_gold',
        'reference_sha256':hashlib.sha256(text.encode()).hexdigest(),
        'inputs_sha256':files,
        'reference_code_and_rubric_sha256':code_hashes,
        'cases':100, 'candidates':1000,
        'independent_initial_ratings':2000,
        'adjudicated_candidates':len(adjudications),
        'known_labels':reference['known_count'], 'unknown_labels':reference['unknown_count'],
        'exact_grade_agreement_count':fresh['exact_grade_agreement_count'],
        'binary_agreement_count_excluding_unknown':fresh['binary_agreement_count_excluding_unknown'],
        'binary_agreement_denominator':fresh['binary_agreement_denominator'],
        'selector_inference_calls':0,
        'new_gpu_jobs':0,
        'annotation_compute_usage':'Codex subagent usage; not metered in this receipt',
        'not_a_human_reference_or_cryptographic_signature':True,
    }
    return text, canonical(receipt)


def verify(root):
    expected_reference, expected_receipt = build(root)
    if (root/'reference.json').read_text() != expected_reference:
        raise ValueError('reference_content_mismatch')
    if (root/'reference-receipt.json').read_text() != expected_receipt:
        raise ValueError('reference_receipt_mismatch')
    return json.loads(expected_receipt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['build','verify'])
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'build':
        reference, receipt = build(args.root)
        print(json.dumps({'reference_text':reference, 'receipt_text':receipt}, ensure_ascii=False))
    else:
        print(json.dumps(verify(args.root), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

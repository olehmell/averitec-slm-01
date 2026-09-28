from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import summarize
from data import EXPERIMENT, ROOT, prepare_selection, selected_cases


def test_selection_is_gold_free_group_disjoint_and_excludes_main_holdout():
    config = yaml.safe_load((EXPERIMENT/'config.yaml').read_text())
    selection = prepare_selection(config)
    assert selection == prepare_selection(config)
    rows = selection['cases']
    assert len(rows) == 120
    assert len({r['group_id'] for r in rows}) == 120
    original = json.loads((ROOT/config['source_split_manifest']).read_text())
    heldout = {case for g in original['groups'] if g['role']=='study_holdout' for case in g['case_ids']}
    assert not heldout.intersection(r['case_id'] for r in rows)
    assert all(set(r)=={'case_id','group_id','split','role'} for r in rows)
    assert len(selected_cases(selection,'development')) == 20
    assert len(selected_cases(selection,'evaluation')) == 100
    corrupted = deepcopy(selection)
    corrupted['cases'][0]['case_id'] = 'averitec-dev-0002'
    with pytest.raises(ValueError, match='selection_not_canonical'):
        selected_cases(corrupted,'development')


def test_summary_keeps_format_modes_separate_and_interrupted_denominator():
    event = {'provider_outcome':'ok','compliant':True,'action':'decompose',
             'expected_action':'decompose','latency_ms':2,'input_tokens':None,
             'output_tokens':None,'observation_sha256':'obs'}
    base = {'controller':'qwen','mode':'checkpoints','scenario':'nominal',
            'order':'canonical','repetition':0,'case_id':'a','group_id':'a',
            'output_mode':'structured','planned_decisions':2,
            'events':[event,event], 'outcome':'checkpoints_evaluated'}
    interrupted = {**base,'case_id':'b','group_id':'b', 'events':[],
                   'outcome':'interrupted_unknown_outcome'}
    prompt = {**base,'output_mode':'prompt_only'}
    result = summarize([base,interrupted,prompt],bootstrap_samples=50)
    assert len(result['arms']) == 2
    structured = next(a for a in result['arms'] if a['output_mode']=='structured')
    assert structured['first_attempt_rule_compliance'] == .5
    assert structured['planned_decisions'] == 4
    assert structured['interrupted_trials'] == 1
    assert structured['token_accounting_missing_calls'] == 2
    unmatched = summarize([base, interrupted, {**base, 'controller': 'lfm'}], bootstrap_samples=50)
    assert unmatched['paired_case_cluster_differences'][0]['status'] == 'unmatched_cases_or_repetitions'


def test_summary_separates_prompt_profile_and_legacy_defaults():
    event = {'provider_outcome': 'ok', 'compliant': True, 'action': 'decompose',
             'expected_action': 'decompose', 'latency_ms': 2, 'input_tokens': 1,
             'output_tokens': 1, 'observation_sha256': 'obs'}
    legacy = {'controller': 'qwen', 'mode': 'checkpoints', 'scenario': 'nominal',
              'order': 'canonical', 'repetition': 0, 'case_id': 'a', 'group_id': 'a',
              'output_mode': 'structured', 'planned_decisions': 1, 'events': [event],
              'outcome': 'checkpoints_evaluated'}
    v2 = {**legacy, 'prompt_variant': 'v2', 'generation_profile': 'qwen_native'}
    result = summarize([legacy, v2], bootstrap_samples=50)
    assert {(arm['prompt_variant'], arm['generation_profile']) for arm in result['arms']} == {
        ('v1', 'baseline'), ('v2', 'qwen_native')}
    assert result['paired_configuration_differences'] == [{
        'controller': 'qwen', 'mode': 'checkpoints', 'scenario': 'nominal', 'output_mode': 'structured',
        'left_prompt_variant': 'v1', 'left_generation_profile': 'baseline',
        'right_prompt_variant': 'v2', 'right_generation_profile': 'qwen_native',
        'paired_groups': 1, 'paired_observations': 1, 'mean_difference': 0.0,
        'bootstrap_95_ci': [0.0, 0.0],
        'interpretation': 'descriptive; configuration contrast, no unadjusted significance claim',
    }]


def test_summary_never_crosses_configuration_for_orders_or_model_pairs():
    event = {'provider_outcome': 'ok', 'compliant': True, 'action': 'decompose',
             'expected_action': 'decompose', 'latency_ms': 2, 'input_tokens': 1,
             'output_tokens': 1, 'observation_sha256': 'obs'}
    base = {'mode': 'checkpoints', 'scenario': 'nominal', 'repetition': 0,
            'case_id': 'a', 'group_id': 'a', 'output_mode': 'structured',
            'planned_decisions': 1, 'events': [event], 'outcome': 'checkpoints_evaluated'}
    rows = [
        {**base, 'controller': 'qwen', 'order': 'canonical', 'prompt_variant': 'v1', 'generation_profile': 'baseline'},
        {**base, 'controller': 'qwen', 'order': 'reversed', 'prompt_variant': 'v2', 'generation_profile': 'qwen_native'},
        {**base, 'controller': 'lfm', 'order': 'canonical', 'prompt_variant': 'v2', 'generation_profile': 'lfm_native'},
    ]
    result = summarize(rows, bootstrap_samples=50)
    assert result['option_order_comparable_checkpoints'] == 0
    assert result['paired_case_cluster_differences'] == []


def test_configuration_pairs_require_matching_group_repetitions():
    event = {'provider_outcome': 'ok', 'compliant': True, 'action': 'decompose',
             'expected_action': 'decompose', 'latency_ms': 2, 'input_tokens': 1,
             'output_tokens': 1, 'observation_sha256': 'obs'}
    base = {'controller': 'lfm', 'mode': 'checkpoints', 'scenario': 'nominal',
            'order': 'canonical', 'case_id': 'a', 'group_id': 'a', 'output_mode': 'structured',
            'planned_decisions': 1, 'events': [event], 'outcome': 'checkpoints_evaluated'}
    result = summarize([
        {**base, 'repetition': 0, 'prompt_variant': 'v1', 'generation_profile': 'baseline'},
        {**base, 'repetition': 1, 'prompt_variant': 'v1', 'generation_profile': 'baseline'},
        {**base, 'repetition': 0, 'prompt_variant': 'v2', 'generation_profile': 'lfm_native'},
    ], bootstrap_samples=50)
    comparison = result['paired_configuration_differences'][0]
    assert comparison['status'] == 'unmatched_cases_or_repetitions'
    assert comparison['paired_groups'] == 1
    assert comparison['paired_observations'] == 1


def test_configuration_bootstrap_clusters_group_means_not_repetitions():
    event = {'provider_outcome': 'ok', 'action': 'decompose', 'expected_action': 'decompose',
             'latency_ms': 2, 'input_tokens': 1, 'output_tokens': 1, 'observation_sha256': 'obs'}

    def row(group, repetition, variant, compliant):
        return {
            'controller': 'qwen', 'mode': 'checkpoints', 'scenario': 'nominal',
            'order': 'canonical', 'case_id': group, 'group_id': group, 'output_mode': 'structured',
            'repetition': repetition, 'prompt_variant': variant, 'generation_profile': 'baseline',
            'planned_decisions': 1, 'events': [{**event, 'compliant': compliant}],
            'outcome': 'checkpoints_evaluated',
        }

    # Group a has two matched repetitions and group b has one.  The arms are
    # balanced within each group, but a must not receive double bootstrap weight.
    uneven = [
        row('a', 0, 'v1', True), row('a', 0, 'v2', False),
        row('a', 1, 'v1', True), row('a', 1, 'v2', False),
        row('b', 0, 'v1', False), row('b', 0, 'v2', True),
    ]
    collapsed = [
        row('a', 0, 'v1', True), row('a', 0, 'v2', False),
        row('b', 0, 'v1', False), row('b', 0, 'v2', True),
    ]
    uneven_comparison = summarize(uneven, bootstrap_samples=100, seed=7)['paired_configuration_differences'][0]
    collapsed_comparison = summarize(collapsed, bootstrap_samples=100, seed=7)['paired_configuration_differences'][0]
    assert uneven_comparison['paired_observations'] == 3
    assert uneven_comparison['paired_groups'] == 2
    assert uneven_comparison['mean_difference'] == collapsed_comparison['mean_difference'] == 0.0
    assert uneven_comparison['bootstrap_95_ci'] == collapsed_comparison['bootstrap_95_ci']

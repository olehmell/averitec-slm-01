"""Descriptive reliability estimates with case-cluster paired differences."""
from __future__ import annotations

from collections import defaultdict
import random
import statistics


def _configuration(row: dict) -> tuple[str, str]:
    """Return the explicit configuration, with stable defaults for v1 records."""
    return row.get('prompt_variant') or 'v1', row.get('generation_profile') or 'baseline'


def _score(row: dict) -> float:
    if row['mode'] == 'checkpoints':
        return sum(event['compliant'] for event in row.get('events', [])) / row['planned_decisions']
    return float(row.get('protocol_correct_termination') is True)


def _paired_difference(left: dict, right: dict, *, seed: int, bootstrap_samples: int,
                       interpretation: str) -> dict:
    """Pair one condition on exactly the same group/repetition observations."""
    left_keys, right_keys = set(left), set(right)
    common = sorted(left_keys & right_keys)
    if left_keys != right_keys:
        return {'paired_groups': len({group for group, _ in common}),
                'paired_observations': len(common), 'mean_difference': None,
                'bootstrap_95_ci': None, 'status': 'unmatched_cases_or_repetitions'}
    by_group = defaultdict(list)
    for group, repetition in common:
        by_group[group].append(left[(group, repetition)] - right[(group, repetition)])
    differences = [statistics.mean(by_group[group]) for group in sorted(by_group)]
    if not differences:
        return {'paired_groups': 0, 'paired_observations': 0, 'mean_difference': None,
                'bootstrap_95_ci': None, 'status': 'no_paired_observations'}
    rng = random.Random(seed)
    samples = sorted(statistics.mean(rng.choices(differences, k=len(differences)))
                     for _ in range(bootstrap_samples))
    return {
        'paired_groups': len({group for group, _ in common}),
        'paired_observations': len(common), 'mean_difference': statistics.mean(differences),
        'bootstrap_95_ci': [samples[int(.025 * bootstrap_samples)],
                             samples[min(bootstrap_samples - 1, int(.975 * bootstrap_samples))]],
        'interpretation': interpretation,
    }


def summarize(rows: list[dict], *, bootstrap_samples=2000, seed=20260918) -> dict:
    arms = defaultdict(list)
    for row in rows:
        prompt_variant, generation_profile = _configuration(row)
        arms[(row['controller'], row['mode'], row['scenario'], row['order'], row['output_mode'],
              prompt_variant, generation_profile)].append(row)
    summary = []
    for (arm, mode, scenario, order, output_mode, prompt_variant, generation_profile), values in sorted(arms.items()):
        events = [event for row in values for event in row.get('events', [])]
        valid = [event for event in events if event['provider_outcome'] == 'ok']
        latency = sorted(event['latency_ms'] for event in events)
        denominator = sum(r['planned_decisions'] for r in values) if mode == 'checkpoints' else len(events)
        # Intent-only interruptions have zero observed decisions; count them at case level.
        result = {
            'controller': arm, 'mode': mode, 'scenario': scenario, 'order': order, 'output_mode': output_mode,
            'prompt_variant': prompt_variant, 'generation_profile': generation_profile,
            'planned_trials': len(values), 'observed_decisions': len(events),
            'planned_decisions': denominator if mode == 'checkpoints' else None,
            'unique_observation_states': len({e['observation_sha256'] for e in events}),
            'first_attempt_rule_compliance': sum(e['compliant'] for e in events) / denominator if denominator else None,
            'compliance_given_valid_output': sum(e['compliant'] for e in valid) / len(valid) if valid else None,
            'schema_failure_rate': sum(e['provider_outcome'] == 'invalid_output' for e in events) / len(events) if events else None,
            'infrastructure_failure_rate': sum(e['provider_outcome'] in ('transport_error', 'version_mismatch') for e in events) / len(events) if events else None,
            'interrupted_trials': sum(r['outcome'] == 'interrupted_unknown_outcome' for r in values),
            'autonomous_full_pipeline_completion': sum(r.get('full_pipeline_completed') is True for r in values) / len(values) if mode == 'trajectory' else None,
            'protocol_correct_termination': sum(r.get('protocol_correct_termination') is True for r in values) / len(values) if mode == 'trajectory' else None,
            'latency_median_ms': statistics.median(latency) if latency else None,
            'latency_p95_ms': latency[min(len(latency)-1, int(0.95*(len(latency)-1)))] if latency else None,
            'measured_input_tokens': sum(e['input_tokens'] for e in events if e['input_tokens'] is not None),
            'measured_output_tokens': sum(e['output_tokens'] for e in events if e['output_tokens'] is not None),
            'token_accounting_missing_calls': sum(e['input_tokens'] is None or e['output_tokens'] is None for e in events),
            'retry_rule_violations': sum(e.get('violation') == 'retry_rule' for e in events),
            'early_finishes': sum(e['action'] == 'finish' and e['expected_action'] != 'finish' for e in events),
        }
        repeated = defaultdict(list)
        for row in values:
            for event in row.get('events', []):
                repeated[(row['case_id'], event['observation_sha256'])].append((event['provider_outcome'], event['action']))
        groups = [v for v in repeated.values() if len(v) >= 2]
        result['repeated_checkpoint_groups'] = len(groups)
        result['decision_repeatability'] = sum(len(set(g)) == 1 for g in groups) / len(groups) if groups else None
        summary.append(result)
    # Compare models only within one prompt/profile configuration.  Pairing uses
    # group and canonical repetition, so a missing repetition never silently
    # changes a contrast's denominator.
    by_model_condition = defaultdict(lambda: defaultdict(dict))
    by_configuration_condition = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if row['order'] != 'canonical' or row['output_mode'] != 'structured':
            continue
        prompt_variant, generation_profile = _configuration(row)
        observation = (row['group_id'], row['repetition'])
        model_condition = (row['mode'], row['scenario'], prompt_variant, generation_profile)
        configuration_condition = (row['controller'], row['mode'], row['scenario'], row['output_mode'])
        by_model_condition[model_condition][row['controller']][observation] = _score(row)
        by_configuration_condition[configuration_condition][(prompt_variant, generation_profile)][observation] = _score(row)
    paired = []
    for condition, controllers in sorted(by_model_condition.items()):
        names = sorted(controllers)
        for i, left in enumerate(names):
            for right in names[i+1:]:
                paired.append({'mode': condition[0], 'scenario': condition[1],
                               'prompt_variant': condition[2], 'generation_profile': condition[3],
                               'left': left, 'right': right,
                               **_paired_difference(controllers[left], controllers[right],
                                                    seed=seed, bootstrap_samples=bootstrap_samples,
                                                    interpretation='descriptive; within-configuration contrast, no unadjusted significance claim')})
    configuration_paired = []
    for condition, configurations in sorted(by_configuration_condition.items()):
        names = sorted(configurations)
        for i, left in enumerate(names):
            for right in names[i+1:]:
                configuration_paired.append({
                    'controller': condition[0], 'mode': condition[1], 'scenario': condition[2],
                    'output_mode': condition[3],
                    'left_prompt_variant': left[0], 'left_generation_profile': left[1],
                    'right_prompt_variant': right[0], 'right_generation_profile': right[1],
                    **_paired_difference(configurations[left], configurations[right],
                                         seed=seed, bootstrap_samples=bootstrap_samples,
                                         interpretation='descriptive; configuration contrast, no unadjusted significance claim'),
                })
    orders = defaultdict(dict)
    for row in rows:
        if row['repetition'] != 0:
            continue
        prompt_variant, generation_profile = _configuration(row)
        for event in row.get('events', []):
            key = (row['controller'], row['mode'], row['scenario'], row['case_id'],
                   event['observation_sha256'], row['output_mode'], prompt_variant, generation_profile)
            orders[key][row['order']] = (event['provider_outcome'], event['action'])
    comparable = [v for v in orders.values() if 'canonical' in v and len(v) > 1]
    return {'arms': summary, 'paired_case_cluster_differences': paired,
            'paired_configuration_differences': configuration_paired,
            'option_order_comparable_checkpoints': len(comparable),
            'option_order_changed_fraction': sum(any(value != v['canonical'] for value in v.values()) for v in comparable)/len(comparable) if comparable else None,
            'quality_evaluation_performed': False}

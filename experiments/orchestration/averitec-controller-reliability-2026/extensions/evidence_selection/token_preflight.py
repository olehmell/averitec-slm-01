"""Exact pinned-tokenizer preflight; no checkpoint weights or model inference.

Downloads only the hash-verified existing Laya wheel and typed tokenizer files
into memory. Tests the production adapter against upstream sequence construction.
"""
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Union
from urllib.request import urlopen
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from selector_runtime import NativeLayaEvidenceSelector


def fetch(url, algorithm, expected):
    with urlopen(url, timeout=30) as response:
        raw = response.read(6_000_001)
    if len(raw) > 6_000_000:
        raise ValueError('asset_size_limit')
    if algorithm == 'sha256':
        actual = hashlib.sha256(raw).hexdigest()
    else:
        actual = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
    if actual != expected:
        raise ValueError('asset_hash_mismatch')
    return raw


def main():
    from tokenizers import Tokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--instructions', type=Path, required=True)
    args = parser.parse_args()
    assets = json.loads(args.assets.read_text())
    package = assets['package']
    wheel = fetch(package['url'], 'sha256', package['sha256'])
    common = zipfile.ZipFile(io.BytesIO(wheel)).read('laya/common.py').decode()
    names = {'serialize_state', 'render_criterion', 'render_options', 'build_sequence'}
    tree = ast.parse(common)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in functions} != names:
        raise ValueError('upstream_sequence_api_changed')
    namespace = dict(json=json, Union=Union, Dict=Dict, List=List, Optional=Optional)
    exec(compile(ast.Module(body=functions, type_ignores=[]), 'verified-laya-common', 'exec'), namespace)
    model = assets['model']
    files = {f['path']: f for f in model['files']}
    loaded = {}
    for name in ('tokenizer.json', 'tokenizer_config.json'):
        relative = 'typed-decisions/tokenizer/' + name
        f = files[relative]
        url = f"https://huggingface.co/{model['repository']}/resolve/{model['revision']}/{relative}"
        loaded[name] = fetch(url, f['digest_algorithm'], f['digest'])
    cfg = json.loads(loaded['tokenizer_config.json'])
    tokenizer = Tokenizer.from_str(loaded['tokenizer.json'].decode())
    tokenizer.no_truncation()
    tokenizer.no_padding()

    class TokenizerAdapter:
        mask_token = cfg['mask_token']
        mask_token_id = tokenizer.token_to_id(cfg['mask_token'])
        cls_token_id = tokenizer.token_to_id(cfg['cls_token'])
        sep_token_id = tokenizer.token_to_id(cfg['sep_token'])

        def __call__(self, text, add_special_tokens=False):
            return {'input_ids': tokenizer.encode(text, add_special_tokens=add_special_tokens).ids}

    class TokenizerOnlyAgent:
        cfg = {'max_len':1024, 'head_max_len':256}
        device = 'cpu'
        tok = TokenizerAdapter()

        def predict(self, *_args, **_kwargs):
            raise AssertionError('inference_forbidden_in_tokenizer_preflight')

    selector = NativeLayaEvidenceSelector(None, device='cpu', _agent=TokenizerOnlyAgent())
    candidates = json.loads(args.candidates.read_text())
    instructions = args.instructions.read_text().strip()
    q = {'t':'choice', 'ins':instructions, 'crit':{'include':None, 'exclude':None}}
    totals, states = [], []
    for case in candidates['cases']:
        for passage in case['candidates']:
            observation = {'claim':case['claim'], 'candidate':{k:passage[k] for k in ('id','text','url')}}
            check = selector.preflight(observation, instructions, ['include','exclude'])
            state = json.dumps(observation, sort_keys=True, separators=(',',':'), ensure_ascii=False)
            actual, markers = namespace['build_sequence'](TokenizerAdapter(), state, q, 1024, 256)
            unbounded, unlimited_markers = namespace['build_sequence'](TokenizerAdapter(), state, q, 100000, 100000)
            if actual != unbounded or markers != unlimited_markers or len(markers) != 2:
                raise ValueError('upstream_sequence_truncation_or_mismatch')
            if len(actual) != check['total_tokens']:
                raise ValueError('adapter_upstream_token_count_mismatch')
            totals.append(len(actual))
            states.append(check['state_tokens'])
    if len(totals) != 1000:
        raise ValueError('planned_input_count')
    print(json.dumps({'schema':'averitec-selector-tokenizer-preflight/v1',
                      'complete':True, 'candidates':len(totals), 'model_forward_calls':0,
                      'package_sha256':package['sha256'], 'model_revision':model['revision'],
                      'candidates_sha256':hashlib.sha256(args.candidates.read_bytes()).hexdigest(),
                      'instructions_sha256':hashlib.sha256(args.instructions.read_bytes()).hexdigest(),
                      'max_total_tokens':max(totals), 'min_total_tokens':min(totals),
                      'median_total_tokens':sorted(totals)[len(totals)//2],
                      'max_state_tokens':max(states), 'context_limit':1024,
                      'upstream_sequence_equality_verified':True,
                      'native_forward_qualified':False}, indent=2))


if __name__ == '__main__':
    main()

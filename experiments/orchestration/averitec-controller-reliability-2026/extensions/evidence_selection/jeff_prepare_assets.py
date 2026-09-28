"""CPU-only preparation in a fresh dedicated directory; never loads weights."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

REVISION = 'd0a4e53d09cebe6bc963dd9be319d4279084bb2d'
MODEL = 'knowledgator/gliformer-large-v1'
FILES = {
    'pytorch_model.bin': ('sha256', 'f80b29199d66f878669f283703e4dba9fd726755dcc20aba1ed0d24fce4a23f1'),
    'gliner_config.json': ('git-sha1', 'e2c308a65d01df736f40cca8c86c33fdaab2cf23'),
    'tokenizer.json': ('git-sha1', 'd9d39f29d1eefec3ad95bd2c71182bb697360bc6'),
    'tokenizer_config.json': ('git-sha1', '4c9378f8e8655946019c966d5f05ff2cfb20b5d5'),
}

def digest(path, algorithm='sha256'):
    h = hashlib.sha256() if algorithm == 'sha256' else hashlib.sha1()
    if algorithm == 'git-sha1': h.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--download-only', action='store_true')
    args = p.parse_args()
    root = args.root.resolve()
    root.mkdir(exist_ok=False)
    checkpoint = root / 'checkpoint'
    checkpoint.mkdir()
    for name, (algorithm, expected) in FILES.items():
        path = checkpoint / name
        url = f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}'
        print(f'Downloading {name}', flush=True)
        with urllib.request.urlopen(url, timeout=90) as source, path.open('xb') as target:
            while block := source.read(1024*1024): target.write(block)
        assert digest(path, algorithm) == expected, f'asset mismatch: {name}'
    if args.download_only:
        wheels = root / 'wheels'
        wheels.mkdir()
        for package, version in [('gliformer', '0.1.2'), ('gliner', '0.2.29'), ('scipy', '1.17.1')]:
            with urllib.request.urlopen(f'https://pypi.org/pypi/{package}/{version}/json', timeout=90) as response:
                metadata = json.load(response)
            options = [f for f in metadata['urls'] if f['filename'].endswith('py3-none-any.whl') or
                       ('cp312-cp312-manylinux' in f['filename'] and 'x86_64' in f['filename'])]
            assert len(options) == 1, (package, [f['filename'] for f in options])
            f = options[0]
            with urllib.request.urlopen(f['url'], timeout=90) as response, (wheels/f['filename']).open('xb') as target:
                while block := response.read(1024*1024): target.write(block)
            assert digest(wheels/f['filename']) == f['digests']['sha256']
        print('Download and digest checks complete; no inference.', flush=True)
        return
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                    '--no-deps', '--target', str(root/'python'),
                    'gliformer==0.1.2', 'gliner==0.2.29', 'scipy==1.17.1'], check=True)
    record = {'model': MODEL, 'model_revision': REVISION,
              'files': {str(f.relative_to(root)): digest(f) for f in sorted(root.rglob('*'))
                        if f.is_file() and '__pycache__' not in f.parts and f.suffix != '.pyc'},
              'base_packages': dict(sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions()))}
    (root/'preparation.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'prepared': str(root), 'files': len(record['files']), 'inference_calls': 0}))

if __name__ == '__main__': main()

"""Seal already staged public Jeff assets; standard library, no inference."""
import argparse
import hashlib
import json
from pathlib import Path

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    files = {}
    for directory in ('checkpoint', 'python', 'jeff-src', 'wheels'):
        assert (args.root/directory).is_dir()
        for path in sorted((args.root/directory).rglob('*')):
            assert not path.is_symlink(), path
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                files[str(path.relative_to(args.root))] = digest(path)
    checkpoint = {k:v for k,v in files.items() if k.startswith('checkpoint/')}
    assert len(checkpoint) == 4
    assert checkpoint['checkpoint/pytorch_model.bin'] == 'f80b29199d66f878669f283703e4dba9fd726755dcc20aba1ed0d24fce4a23f1'
    expected_wheels = {'gliformer-0.1.2-py3-none-any.whl': 'b9a1194b56fc0ab193fef919ad68d0b3086ce2fe3f0b4392ce26873139ac8768',
                       'gliner-0.2.29-py3-none-any.whl': '0c8cfb9f5c2daf7aff329ebb5ab1609052d7ad0aca0ef26049b9476b018b3fde',
                       'scipy-1.17.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl': '02ae3b274fde71c5e92ac4d54bc06c42d80e399fec704383dcd99b301df37458'}
    for name, expected in expected_wheels.items(): assert files['wheels/'+name] == expected
    record = {'schema': 'averitec-jeff-native-assets/v1',
              'upstream_commit': '34b32f99a727c47b679adde33f4702a001e02979',
              'model': 'knowledgator/gliformer-large-v1',
              'model_revision': 'd0a4e53d09cebe6bc963dd9be319d4279084bb2d',
              'checkpoint_sha256': hashlib.sha256(json.dumps(checkpoint, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
              'base_image_sha256': '88e35b0554e3d1da16dfe8b9944e866adde00566935b69b366acf83d32678c08',
              'files': files}
    with args.output.open('x') as f: json.dump(record, f, indent=2, sort_keys=True)
    print(json.dumps({'asset_manifest_sha256': digest(args.output), 'file_count': len(files), 'checkpoint_sha256': record['checkpoint_sha256']}))

if __name__ == '__main__': main()

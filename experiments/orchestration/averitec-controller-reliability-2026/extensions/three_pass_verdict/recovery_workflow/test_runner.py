"""Offline checks of the durable provider boundary; no network calls."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import URLError
import unittest

MODULE = Path(__file__).with_name('runner.py')
spec = importlib.util.spec_from_file_location('recovery_runner', MODULE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

LIMITS = {'maximum_total_generation_requests': 2301,
          'maximum_pass_generation_requests': 2300,
          'maximum_measured_generation_requests': 2299,
          'maximum_warmup_generation_requests': 1,
          'maximum_smoke_generation_requests': 1,
          'maximum_input_tokens': 100, 'maximum_output_tokens': 100,
          'maximum_wall_seconds': 100}


class BoundaryTests(unittest.TestCase):
    def test_intent_precedes_response_and_caps_count(self):
        with TemporaryDirectory() as directory:
            ledger = Path(directory) / 'ledger.jsonl'
            original = runner.providers._http_post
            calls = []
            try:
                def mock(url, body, headers, timeout):
                    calls.append(1)
                    self.assertEqual(json.loads(ledger.read_text().splitlines()[-1])['kind'], 'intent')
                    return {'model': 'jev-1.13.0', 'usage': {'input_tokens': 2, 'output_tokens': 1}}
                runner.providers._http_post = mock
                with runner.metered_http('jev', ledger, LIMITS, phase='smoke', smoke_used=0) as counter:
                    runner.providers._http_post('https://example.invalid', {'a': 1}, {}, 1)
                    with self.assertRaises(runner.StopRun):
                        runner.providers._http_post('https://example.invalid', {'a': 1}, {}, 1)
                self.assertEqual(len(calls), 1)
                self.assertEqual(counter['requests'], 1)
                self.assertEqual([json.loads(x)['kind'] for x in ledger.read_text().splitlines()], ['intent', 'response'])
            finally:
                runner.providers._http_post = original

    def test_transport_leaves_uncertain_and_stops(self):
        with TemporaryDirectory() as directory:
            ledger = Path(directory) / 'ledger.jsonl'
            original = runner.providers._http_post
            try:
                def mock(*_args):
                    raise URLError('network')
                runner.providers._http_post = mock
                with runner.metered_http('jev', ledger, LIMITS, phase='pass', smoke_used=1):
                    with self.assertRaises(runner.StopRun):
                        runner.providers._http_post('https://example.invalid', {}, {}, 1)
                self.assertEqual([json.loads(x)['kind'] for x in ledger.read_text().splitlines()], ['intent', 'uncertain'])
            finally:
                runner.providers._http_post = original

    def test_identity_drift_stops_after_known_response(self):
        with TemporaryDirectory() as directory:
            ledger = Path(directory) / 'ledger.jsonl'
            original = runner.providers._http_post
            try:
                runner.providers._http_post = lambda *_: {'model': 'other', 'usage': {'input_tokens': 1, 'output_tokens': 1}}
                with runner.metered_http('jev', ledger, LIMITS, phase='pass', smoke_used=1):
                    with self.assertRaises(runner.StopRun):
                        runner.providers._http_post('https://example.invalid', {}, {}, 1)
                self.assertEqual([json.loads(x)['kind'] for x in ledger.read_text().splitlines()], ['intent', 'response'])
            finally:
                runner.providers._http_post = original


if __name__ == '__main__':
    unittest.main()

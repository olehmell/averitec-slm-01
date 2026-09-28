"""Offline accounting and retry checks. No provider access."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from urllib.error import HTTPError, URLError

import runner
import providers


class Response:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps({"model": runner.MODEL,
                           "usage": {"input_tokens": 10, "output_tokens": 2}}).encode()


class Opener:
    def __init__(self, outcomes):
        self.outcomes = outcomes

    def open(self, *_args, **_kwargs):
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def counters():
    return {"logical_calls": 1, "http_attempts": 0, "extra_attempts": 0,
            "input_charged": 0, "output_charged": 0, "first_attempt_529": 0,
            "started": 0, "last_attempt_start": None}


def manifest(phase="pass"):
    return {"phase": phase, "limits": {"http_attempts": 1101 if phase == "pass" else 1,
                                        "extra_http_attempts": 100 if phase == "pass" else 0,
                                        "input_tokens": 10000, "output_tokens": 10000,
                                        "wall_seconds": 1000,
                                        "minimum_request_spacing_seconds": 1 if phase == "pass" else 0}}


class MeteredTests(unittest.TestCase):
    def exercise(self, phase, outcomes):
        body = {"model": runner.MODEL, "state": {"x": 1}}
        original = runner.build_opener
        runner.build_opener = lambda *_args: Opener(outcomes)
        try:
            with TemporaryDirectory() as directory:
                ledger = Path(directory) / "ledger.jsonl"
                ledger.touch()
                count = counters()
                waits = []
                with runner.metered_http(m=manifest(phase), ledger=ledger,
                                         expected_sha=runner.body_sha(body), ordinal=2,
                                         counters=count, sleep=waits.append, clock=lambda: 0):
                    result = providers._http_post(providers.TYPESAFE_SYSTEM_ONE_URL, body, {}, 30)
                events = [json.loads(line) for line in ledger.read_text().splitlines()]
                return result, events, count, waits
        finally:
            runner.build_opener = original

    def test_explicit_529_retries_same_payload_with_two_intents(self):
        error = HTTPError(providers.TYPESAFE_SYSTEM_ONE_URL, 529, "rate limit", {}, None)
        _, events, count, waits = self.exercise("pass", [error, Response()])
        self.assertEqual([event["kind"] for event in events],
                         ["intent", "http_error", "retry_wait", "intent", "response"])
        self.assertEqual(events[0]["request_sha256"], events[3]["request_sha256"])
        self.assertEqual((count["http_attempts"], count["extra_attempts"], count["first_attempt_529"]), (2, 1, 1))
        self.assertEqual((count["input_charged"], count["output_charged"]),
                         (events[0]["reserved_input_tokens"] + 10, 1026))
        self.assertGreaterEqual(waits[0], 5)

    def test_smoke_529_has_no_retry(self):
        original = runner.build_opener
        runner.build_opener = lambda *_args: Opener([HTTPError(providers.TYPESAFE_SYSTEM_ONE_URL, 529, "rate limit", {}, None)])
        try:
            with TemporaryDirectory() as directory:
                ledger = Path(directory) / "ledger.jsonl"
                ledger.touch()
                body = {"x": 1}
                with runner.metered_http(m=manifest("smoke"), ledger=ledger,
                                         expected_sha=runner.body_sha(body), ordinal=2,
                                         counters=counters(), sleep=lambda _: self.fail("retry"), clock=lambda: 0):
                    with self.assertRaisesRegex(runner.StopRun, "http_529"):
                        providers._http_post(providers.TYPESAFE_SYSTEM_ONE_URL, body, {}, 30)
                self.assertEqual([json.loads(x)["kind"] for x in ledger.read_text().splitlines()],
                                 ["intent", "http_error"])
        finally:
            runner.build_opener = original

    def test_unknown_transport_stops_without_retry(self):
        original = runner.build_opener
        runner.build_opener = lambda *_args: Opener([URLError("timeout")])
        try:
            with TemporaryDirectory() as directory:
                ledger = Path(directory) / "ledger.jsonl"
                ledger.touch()
                body = {"x": 1}
                with runner.metered_http(m=manifest(), ledger=ledger,
                                         expected_sha=runner.body_sha(body), ordinal=2,
                                         counters=counters(), sleep=lambda _: self.fail("retry"), clock=lambda: 0):
                    with self.assertRaisesRegex(runner.StopRun, "unknown_transport_or_decode"):
                        providers._http_post(providers.TYPESAFE_SYSTEM_ONE_URL, body, {}, 30)
                self.assertEqual([json.loads(x)["kind"] for x in ledger.read_text().splitlines()],
                                 ["intent", "uncertain"])
        finally:
            runner.build_opener = original


if __name__ == "__main__":
    unittest.main()

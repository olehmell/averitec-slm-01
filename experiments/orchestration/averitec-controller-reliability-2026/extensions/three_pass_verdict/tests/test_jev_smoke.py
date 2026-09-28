"""The single hosted smoke must stop slow requests and reject redirects."""

from __future__ import annotations

from pathlib import Path
import signal
import sys
import time

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jev_smoke as smoke  # noqa: E402


def test_deadline_interrupts_and_restores_transport() -> None:
    original = smoke.runner.provider_module._http_post
    with pytest.raises(TimeoutError, match="wall_deadline"):
        with smoke._single_request_deadline(0.05):
            assert smoke.runner.provider_module._http_post is smoke._one_post_no_redirect
            time.sleep(0.2)
    assert smoke.runner.provider_module._http_post is original
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_redirect_handler_refuses_every_redirect() -> None:
    handler = smoke._NoRedirect()
    assert handler.redirect_request(None, None, 302, "found", {}, "https://other.example") is None

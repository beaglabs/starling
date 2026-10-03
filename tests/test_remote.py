import json

import pytest

from starlings import RemoteFactor


class Response:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, _limit):
        return self.payload


def result(version_id="fver_1"):
    return {
        "result": {
            "factorId": "fac_1",
            "factorName": "technical-readiness",
            "versionId": version_id,
            "version": 3,
            "modelId": "a" * 64,
            "outcome": "medium",
            "probabilities": {"low": 0.1, "medium": 0.8, "high": 0.1},
            "confidence": 0.8,
            "calibrated": True,
            "requiresReview": False,
            "reasons": [],
            "provenance": {"pipeline_hash": "b" * 64},
        }
    }


def test_remote_factor_calls_versioned_papyrus_endpoint_and_normalizes_result():
    observed = {}

    def opener(request, timeout):
        observed["url"] = request.full_url
        observed["headers"] = dict(request.header_items())
        observed["body"] = json.loads(request.data)
        observed["timeout"] = timeout
        return Response(result())

    factor = RemoteFactor(
        "https://papyrus.example",
        "technical-readiness",
        token="secret-token",
        expected_version_id="fver_1",
        opener=opener,
    )
    value = factor.evaluate(content="Verification evidence", state={"phase": "PDR"}, threshold=0.9)

    assert observed["url"] == "https://papyrus.example/api/v1/factors/technical-readiness/evaluate"
    assert observed["body"] == {
        "content": "Verification evidence",
        "state": {"phase": "PDR"},
        "threshold": 0.9,
    }
    assert observed["headers"]["Authorization"] == "Bearer secret-token"
    assert value.factor_id == "fac_1"
    assert value.version_id == "fver_1"
    assert value.outcome == "medium"
    assert value.probabilities["medium"] == 0.8
    assert value.calibrated and not value.requires_review


def test_remote_factor_can_pin_version_identity():
    factor = RemoteFactor(
        "https://papyrus.example",
        "technical-readiness",
        expected_version_id="fver_expected",
        opener=lambda *_args, **_kwargs: Response(result("fver_changed")),
    )
    with pytest.raises(RuntimeError, match="version changed"):
        factor.evaluate(content="Evidence")


def test_remote_factor_requires_https_outside_loopback_and_valid_probability_contract():
    with pytest.raises(ValueError, match="HTTPS"):
        RemoteFactor("http://papyrus.example", "risk")

    bad = result()
    bad["result"]["probabilities"] = {"low": 0.2, "medium": 0.2, "high": 0.2}
    factor = RemoteFactor(
        "http://localhost:3210",
        "risk",
        opener=lambda *_args, **_kwargs: Response(bad),
    )
    with pytest.raises(RuntimeError, match="sum to one"):
        factor.evaluate(content="Evidence")

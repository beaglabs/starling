"""Remote Factor client for a governed Papyrus Factor endpoint."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from pydantic import Field, model_validator

from .schemas import StrictModel


class RemoteFactorResult(StrictModel):
    """Version-pinned result returned by a Papyrus ``papyrus.factor.v1`` endpoint."""

    factor_id: str
    factor_name: str
    version_id: str
    version: int = Field(ge=1)
    model_id: str | None = None
    outcome: str
    probabilities: dict[str, float]
    confidence: float = Field(ge=0, le=1)
    calibrated: bool
    requires_review: bool
    reasons: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_probabilities(self) -> RemoteFactorResult:
        if not self.probabilities:
            raise ValueError("Remote Factor result contains no probabilities")
        if any(
            not math.isfinite(value) or not 0 <= value <= 1 for value in self.probabilities.values()
        ):
            raise ValueError("Remote Factor probabilities must be finite and in [0,1]")
        total = sum(self.probabilities.values())
        if abs(total - 1) > 1e-4:
            raise ValueError("Remote Factor probabilities must sum to one")
        if self.outcome not in self.probabilities:
            raise ValueError("Remote Factor outcome is not present in probabilities")
        return self


class RemoteFactor:
    """Call a published Papyrus Factor without embedding its local checkpoint.

    The endpoint remains authoritative for the promoted model version. Every
    result includes that immutable version identity so callers can record or
    reject version changes instead of silently consuming a moving classifier.
    """

    def __init__(
        self,
        base_url: str,
        factor: str,
        *,
        token: str | None = None,
        timeout: float = 30.0,
        expected_version_id: str | None = None,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme not in {"https", "http"} or not parsed.hostname:
            raise ValueError("base_url must be an HTTP(S) origin")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain credentials, query, or fragment")
        if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Remote Factors require HTTPS outside loopback development")
        if not factor.strip():
            raise ValueError("factor is required")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive")
        self.base_url = base_url.rstrip("/")
        self.factor = factor.strip()
        self.token = token
        self.timeout = timeout
        self.expected_version_id = expected_version_id
        self._opener = opener

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/api/v1/factors/{quote(self.factor, safe='')}/evaluate"

    def evaluate(
        self,
        *,
        content: str | dict[str, Any],
        state: dict[str, Any] | None = None,
        threshold: float | None = None,
    ) -> RemoteFactorResult:
        if isinstance(content, str) and not content.strip():
            raise ValueError("content must not be empty")
        if not isinstance(content, (str, dict)):
            raise ValueError("content must be text or a Starlings Content object")
        if state is not None and not isinstance(state, dict):
            raise ValueError("state must be a JSON object")
        payload: dict[str, Any] = {"content": content, "state": state or {}}
        if threshold is not None:
            if not math.isfinite(threshold) or not 0.5 <= threshold <= 1:
                raise ValueError("threshold must be in [0.5,1]")
            payload["threshold"] = threshold

        data = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
        headers = {"accept": "application/json", "content-type": "application/json"}
        if self.token:
            headers["authorization"] = f"Bearer {self.token}"
        request = Request(self.endpoint, data=data, headers=headers, method="POST")
        try:
            with self._opener(request, timeout=self.timeout) as response:
                raw = response.read(8 * 1024 * 1024 + 1)
                if len(raw) > 8 * 1024 * 1024:
                    raise RuntimeError("Remote Factor response exceeded 8 MiB")
        except HTTPError as exc:
            detail = _failure_detail(exc)
            raise RuntimeError(f"Remote Factor request failed ({exc.code}): {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Remote Factor endpoint is unavailable: {exc.reason}") from exc

        try:
            envelope = json.loads(raw)
            if not isinstance(envelope, dict) or not isinstance(envelope.get("result"), dict):
                raise ValueError("response does not contain a result object")
            result = _normalize_result(envelope["result"])
            parsed = RemoteFactorResult.model_validate(result)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            raise RuntimeError(f"Remote Factor returned an invalid response: {exc}") from exc

        if self.expected_version_id and parsed.version_id != self.expected_version_id:
            raise RuntimeError(
                f"Remote Factor version changed: expected {self.expected_version_id}, "
                f"observed {parsed.version_id}"
            )
        return parsed


def _normalize_result(value: dict[str, Any]) -> dict[str, Any]:
    mapping = {
        "factorId": "factor_id",
        "factorName": "factor_name",
        "versionId": "version_id",
        "modelId": "model_id",
        "requiresReview": "requires_review",
    }
    return {mapping.get(key, key): item for key, item in value.items()}


def _failure_detail(error: HTTPError) -> str:
    try:
        raw = error.read(64 * 1024)
        payload = json.loads(raw)
        if isinstance(payload, dict) and isinstance(payload.get("error"), str):
            return payload["error"][:2048]
        return raw.decode(errors="replace")[:2048]
    except Exception:
        return str(error.reason)

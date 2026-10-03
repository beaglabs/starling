# Remote Papyrus Factors

Starlings can run a checkpoint locally with `Classifier`, or call a Factor that an organization has human-promoted and published through Papyrus.

```python
from starlings import RemoteFactor

factor = RemoteFactor(
    "https://papyrus.example.mil",
    "technical-readiness",
    token="...",
)

result = factor.evaluate(
    content="Verification evidence for the current design baseline.",
    state={"phase": "PDR"},
    threshold=0.9,
)

print(result.outcome)
print(result.probabilities)
print(result.version_id)
```

`RemoteFactor` uses the Papyrus `papyrus.factor.v1` endpoint:

```text
POST /api/v1/factors/{factor}/evaluate
```

The request preserves the same Starlings concepts used by local inference: source `content`, caller-supplied structured `state`, and the Factor's server-governed question/alternative contract. The response includes the exact Factor version, model identity when available, per-label probabilities, confidence, calibration state, review requirement, reasons, and provenance.

## Version pinning

A caller can fail closed if the published model changes:

```python
factor = RemoteFactor(
    "https://papyrus.example.mil",
    "technical-readiness",
    expected_version_id="fver_abc123",
)
```

If Papyrus later promotes and publishes another version, inference raises rather than silently accepting the replacement. This lets a Decision evaluation or downstream application explicitly control when it adopts a new learned Factor.

## Transport security

Remote origins require HTTPS. Plain HTTP is accepted only for loopback development (`localhost`, `127.0.0.1`, or `::1`). Credentials are sent in the `Authorization: Bearer ...` header and are never accepted in the base URL.

The remote client does not change Papyrus governance. Training, evaluation, human promotion, publication, and model provenance remain server-side responsibilities; `RemoteFactor` is only an inference client.

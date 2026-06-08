# L13B - Structured Output and Evals

## Why structured output matters

A fluent answer is not enough for production systems. Many applications need typed, machine-readable, validated output:

- invoice extraction
- tool calls
- database writes
- medical or legal triage
- retrieval citations
- workflow state updates

If the model returns malformed JSON, the product breaks even if the prose "looks right."

## 1. Schema as contract

Example invoice schema:

```json
{
  "type": "object",
  "required": ["invoice_id", "total", "currency", "risk_flags"],
  "properties": {
    "invoice_id": {"type": "string"},
    "total": {"type": "number"},
    "currency": {"enum": ["USD", "EUR", "INR"]},
    "risk_flags": {
      "type": "array",
      "items": {"type": "string"}
    }
  }
}
```

The schema does three jobs:

1. It tells the model what shape to produce.
2. It tells the decoder what tokens are legal.
3. It tells the application how to validate the result.

## 2. Logit biasing

Logit biasing changes token probabilities before sampling.

Use it to:

- discourage unwanted phrases
- encourage exact labels
- reduce likelihood of invalid enums

Do not treat it as a guarantee. It is a soft steering mechanism.

## 3. Guided decoding

Guided decoding uses a grammar, JSON schema, regex, or finite-state machine to mask impossible next tokens.

Example:

```text
Partial output:
{"currency": "

Allowed next tokens:
USD
EUR
INR

Disallowed:
dollars
rupees
N/A
newline
```

The sampler sets disallowed tokens to negative infinity before sampling.

Guided decoding gives syntactic guarantees, but it does not guarantee semantic correctness. The model can output valid JSON with the wrong total.

## 4. Validation pipeline

Use layered checks:

```text
raw output
  -> parse JSON
  -> validate JSON schema
  -> validate business rules
  -> validate semantic consistency
  -> run safety filters
  -> return or repair/fallback
```

Example business rules:

- total must be non-negative
- currency must match invoice country
- date must be in acceptable range
- cited document spans must exist
- tool call arguments must match tool permissions

## 5. Eval harnesses

An eval harness is CI for model behavior.

Minimum record:

```yaml
id: invoice_042
input: "raw invoice text..."
expected:
  invoice_id: INV-2026-042
  total: 1287.50
  currency: USD
checks:
  - parse_json
  - schema_valid
  - exact_match.invoice_id
  - numeric_tolerance.total <= 0.01
  - enum.currency
  - latency_budget.p99_ms <= 1200
```

Eval dimensions:

- syntax validity
- field-level accuracy
- semantic correctness
- refusal correctness
- safety behavior
- latency
- cost
- output length

## 6. Benchmarking methodology

Do not report a single "latency" number. Break it down:

- network time
- queue time
- tokenization time
- prefill time
- TTFT
- inter-token latency
- total generation time
- validation time
- guardrail time
- post-processing time

Report percentiles, not just means:

```text
P50: normal user
P90: stressed user
P99: the user who files the ticket
```

## 7. Release gates

A model, prompt, quantization, or guided-decoding change should pass:

```text
json_validity >= baseline
field_accuracy >= baseline - tolerance
safety_rate <= baseline + guard_band
P99_latency <= baseline * 1.10
cost_per_request <= baseline * 1.15
```

This ties structured output back to production systems. Schema validity, latency, and cost are serving metrics.


# L13A - Production Inference Systems

## Why this lecture exists

Earlier lectures taught how a single request becomes tokens quickly:

- prefill and decode
- KV cache
- FlashAttention
- PagedAttention
- prefix caching
- chunked prefill
- quantization
- continuous batching
- speculative decoding
- tensor and pipeline parallelism

Production systems ask a different question:

> How do we turn those optimizations into a service that survives real users, bursty traffic, deploys, safety requirements, and cost pressure?

The answer is that inference engineering becomes a control system. The gateway controls who can call the model. The router controls which model and replica should serve a request. The scheduler controls batching and memory. The deployment system controls canaries and rollbacks. The observability stack controls whether we can understand failures before the bill or the customer does.

## 1. Cold starts

A cold start is not just a slow container. For LLM serving it usually includes:

1. GPU allocation.
2. Container startup.
3. CUDA context creation.
4. Model weight download or local disk read.
5. Weight load into HBM.
6. Runtime initialization.
7. Kernel warmup or CUDA graph capture.
8. KV cache block pool allocation.
9. Tokenizer and request path warmup.

For a 70B model in FP16, the weight memory alone is around 140 GB. Even with quantization, moving weights into the right place is expensive. This is why "scale from zero" is usually wrong for interactive LLMs.

### Warm-pool strategy

Keep a small pool of loaded replicas per model tier:

```text
expected burst capacity = p95 requests/sec over warmup window
warm replicas = ceil(expected burst tokens/sec / safe_tokens_per_replica)
```

Scale based on:

- queue depth
- active sequences
- active tokens
- GPU memory pressure
- TTFT P99
- decode ITL
- tokens/sec demand

Do not scale primarily on CPU utilization. A GPU inference server can be drowning while CPU usage looks calm.

## 2. Canary deployments

LLM canaries must measure more than errors and latency. A model or prompt change can silently alter:

- average output length
- refusal rate
- schema validity
- safety filter rate
- tool call shape
- citation behavior
- cost per request
- prefix cache hit rate

Recommended rollout:

1. Offline evals.
2. Shadow traffic.
3. 1 percent low-risk canary.
4. 5 percent canary.
5. 25 percent canary.
6. Full rollout.

Promotion gate:

```text
quality_score >= baseline - tolerance
schema_validity >= baseline
P99_TTFT <= baseline * 1.10
P99_ITL <= baseline * 1.10
avg_output_tokens <= baseline * 1.15
safety_flag_rate <= baseline + guard_band
cost_per_request <= baseline * 1.15
```

## 3. Cache-aware routing

Standard load balancing routes to the least loaded replica. LLM serving often needs cache-aware routing.

If request A and request B share a 9k-token prefix, request B should often go to the replica that already has request A's prefix KV blocks.

Decision rule:

```text
choose cache-hit replica if:

queue_delay_cache_hit + residual_prefill
<
queue_delay_empty_replica + full_prefill
```

This ties directly to prefix caching and RadixAttention. Routing is no longer just about load. It is about the cost of recomputing tokens.

## 4. Guardrails and output filtering

Guardrails sit around inference:

```text
request
  -> auth and quota
  -> input classifier
  -> prompt injection checks
  -> model inference
  -> structured output validation
  -> safety classifier
  -> business-rule validation
  -> response or fallback
```

Use cheap deterministic checks first. Use expensive judge models only where needed: high-risk tasks, sampled auditing, or escalation.

## 5. Batch vs online inference

Online inference optimizes user-perceived latency:

- low TTFT
- stable ITL
- streaming
- bounded queue time

Batch inference optimizes throughput:

- large batches
- high GPU utilization
- retry and resume
- lower cost per token

Never let unbounded batch jobs share the same priority queue as online decode. Batch work can use idle capacity, but it must yield to realtime traffic.

## Final mental model

Production inference is a set of controlled trade-offs:

- warm pools trade cost for latency
- batching trades latency for throughput
- cache-aware routing trades load balance for less prefill
- canaries trade rollout speed for safety
- guardrails trade latency for correctness and policy compliance
- batch lanes trade freshness for efficiency


# L13D - Final System Design Interview Answer

Prompt:

> Design a low-latency, high-throughput LLM inference system handling millions of requests. Walk me through the engineering trade-offs.

## 1. Clarify the workload

Start with:

- model size and number of model tiers
- prompt length distribution
- output length distribution
- request rate by region
- concurrency
- streaming vs non-streaming
- TTFT SLO
- inter-token latency SLO
- total latency SLO
- structured output requirements
- safety requirements
- batch vs online mix

Millions of requests is not enough information. Tokens/sec and context length distribution drive the architecture.

## 2. High-level architecture

```text
client
  -> regional edge
  -> API gateway
  -> auth, quota, rate limit
  -> input guardrails
  -> prompt normalization
  -> inference router
       model tier selection
       online vs batch selection
       cache-aware replica selection
       canary/stable routing
       fallback provider routing
  -> serving engine
       prefill
       decode
       stream
  -> output validation
  -> safety filter
  -> telemetry
```

## 3. Serving engine

Use a vLLM, TensorRT-LLM, or SGLang-style engine with:

- PagedAttention for KV cache memory management
- continuous batching for high utilization
- chunked prefill to protect decode
- prefix caching for repeated prompts
- speculative decoding where draft acceptance is high
- quantization when evals show quality holds
- tensor parallelism for large model replicas

## 4. Scaling strategy

Use replica parallelism first for throughput. Use tensor parallelism when the model does not fit or one GPU cannot deliver enough bandwidth. Keep tensor parallelism inside high-bandwidth NVLink domains when possible.

Use pipeline parallelism across nodes only when model capacity forces it, because inter-node communication is expensive.

Use disaggregated prefill/decode serving when long prefill work hurts decode SLOs. Account for KV transfer time.

## 5. Routing strategy

Router inputs:

- prompt length
- expected output length
- model difficulty score
- tenant tier
- prefix hash
- cache hit estimate
- replica queue depth
- canary policy
- region and data boundary

Route to the cache-hit replica if the saved prefill time beats the extra queue delay.

## 6. Production safety

Use:

- offline evals
- shadow traffic
- canary rollout
- structured output validation
- output filtering
- rollback
- per-tenant rate limits
- abuse detection

Track:

- TTFT
- ITL
- total latency
- queue depth
- GPU utilization
- VRAM
- KV cache hit rate
- output token distribution
- schema validity
- safety flags
- cost per request

## 7. Engineering trade-offs

| Choice | Benefit | Cost |
|---|---|---|
| Larger batches | higher throughput | higher tail latency |
| Warm pools | lower cold starts | idle GPU cost |
| Cache-aware routing | lower prefill | load imbalance |
| Quantization | lower memory and cost | quality risk |
| Speculative decoding | lower latency | draft model complexity |
| Disaggregated serving | protects decode | KV transfer tax |
| Guided decoding | valid JSON | sampler overhead |
| Strict guardrails | safer outputs | extra latency |

## 8. Final spoken answer

I would start by clarifying workload shape: prompt lengths, output lengths, request rate, concurrency, model size, streaming expectations, and SLOs for TTFT, inter-token latency, total latency, and availability. Millions of requests is not the sizing unit. Tokens/sec and context length distribution are.

I would place a regional gateway in front of the system for auth, quotas, rate limits, request normalization, and input safety checks. Behind that I would use a smart inference router that chooses the model tier, decides online versus batch, routes cache-sensitive requests to replicas with useful prefix KV state, applies canary policy, and uses fallback providers during overload or outages.

For online serving I would use a modern inference engine with PagedAttention, continuous batching, prefix caching, chunked prefill, and streaming decode. I would use quantization only after evals verify quality. I would add speculative decoding if a cheap draft model has a high acceptance rate for the workload.

For scaling I would replicate model servers for throughput, use tensor parallelism inside NVLink domains for large models, and use pipeline parallelism across nodes only when necessary. If long prompts interfere with decode SLOs, I would separate prefill and decode pools, but only after comparing the saved decode delay with the KV transfer tax.

I would keep batch inference on a separate priority lane. Batch jobs should maximize GPU utilization and can tolerate queueing, while online traffic needs bounded TTFT and stable token streaming.

Every release goes through offline evals, shadow traffic, canary rollout, and rollback gates. The production dashboard tracks TTFT, ITL, queue depth, GPU utilization, VRAM, KV cache hit rate, output-token distribution, schema validity, safety flags, and cost per request.

The main trade-off is latency versus utilization. Batching, caching, routing, warm pools, and disaggregation are all tools for choosing that trade-off per request class instead of globally.


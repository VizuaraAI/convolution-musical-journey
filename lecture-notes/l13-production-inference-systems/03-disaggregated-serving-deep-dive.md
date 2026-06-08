# L13C - Disaggregated Serving Deep Dive

## Core idea

Autoregressive inference has two workloads:

- prefill: process all prompt tokens and build KV cache
- decode: generate one token per active request per iteration

Prefill is wide and compute-heavy. Decode is narrow, repeated, and memory-bandwidth-sensitive. Disaggregated serving separates them into different pools.

## Why colocated serving struggles

In a colocated engine, the same GPU alternates between:

1. processing large prefill chunks from new requests
2. generating decode tokens for existing requests

Long prefill bursts can delay decode iterations. Users experience this as uneven streaming or high inter-token latency.

## Disaggregated architecture

```text
request
  -> router
  -> prefill pool
       compute prompt KV
       send KV to decode pool
  -> decode pool
       append one token at a time
       stream output
```

The system has to pay a KV transfer tax.

## KV transfer size

Approximate KV cache bytes:

```text
KV bytes = tokens
         * layers
         * 2
         * n_kv_heads
         * head_dim
         * bytes_per_element
```

For a GQA model:

- prompt tokens = 8192
- layers = 80
- n_kv_heads = 8
- head_dim = 128
- bytes = 2

```text
KV bytes = 8192 * 80 * 2 * 8 * 128 * 2
         = 2,684,354,560 bytes
         ~= 2.5 GiB
```

Moving 2.5 GiB is not free. Disaggregation helps only if the saved interference is larger than the transfer overhead.

## When disaggregation helps

Good fit:

- long prompts
- many active decode streams
- strict ITL SLO
- prefill bursts
- separate hardware pools available
- high-speed interconnect

Bad fit:

- short prompts
- tiny output lengths
- weak network
- low traffic
- simple single-replica deployments

## Sizing intuition

Prefill capacity depends on input tokens/sec.

Decode capacity depends on active sequences and output tokens/sec.

If traffic has long prompts and short outputs, the system needs more prefill capacity. If traffic has short prompts and long outputs, it needs more decode capacity.

## Relationship to earlier lectures

- PagedAttention makes the KV memory manageable.
- Prefix caching reduces prefill work before disaggregation is needed.
- Chunked prefill is the first defense against prefill/decode interference.
- Tensor parallelism is still used inside each prefill or decode replica.
- Quantization reduces weight memory and bandwidth pressure.
- Observability tells whether the split is working: TTFT should improve without ITL regression.

## Decision rule

Use disaggregated serving if:

```text
saved_decode_delay + saved_prefill_queueing
>
KV_transfer_time + orchestration_overhead
```

That is the entire trade-off.


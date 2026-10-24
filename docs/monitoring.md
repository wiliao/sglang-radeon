# Monitoring SGLang on RDNA3

SGLang has no monitoring UI of its own: the server root answers
`"SGLang is running"` and `/docs` is only the OpenAPI page. What it does have is
a Prometheus endpoint plus a Grafana dashboard in
`examples/monitoring/`, and both need one adjustment on this hardware.

## Metrics

Start the server with `--enable-metrics` or `/metrics` is a 404:

```
--enable-metrics
```

That exposes ~62 metric families, all prefixed `sglang:` — with a **colon**:

```
sglang:gen_throughput
sglang:prompt_tokens_total
sglang:generation_tokens_total
sglang:time_to_first_token_seconds_bucket
sglang:inter_token_latency_seconds_bucket
sglang:num_running_reqs
```

## The bundled dashboard renders blank

Imported as-is, every panel is empty, for two independent reasons:

1. **Datasource uid.** Panels target `ddyfngn31dg5cf`, the uid from the
   dashboard author's Grafana. Yours is whatever Grafana assigned at
   provisioning, so the panels have nothing to query.
2. **Metric names.** Queries use `sglang_gen_throughput`; the server emits
   `sglang:gen_throughput`. The colon survives exposition — it is not rewritten
   to an underscore — so they match nothing. 14 expressions are affected.

A panel with no series looks identical to a panel with a bug, so use the script
rather than fixing by hand:

```bash
UID=$(curl -s -u admin:admin localhost:3001/api/datasources \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)[0]["uid"])')
python3 scripts/fix_grafana_dashboard.py \
  --input  <sglang>/examples/monitoring/grafana/dashboards/json/sglang-dashboard.json \
  --output ./sglang-dashboard.fixed.json --datasource-uid "$UID"
```

Point Grafana's file provider at the output, or import it through the UI.

## The four panels upstream is missing

The upstream dashboard covers end-to-end latency, TTFT, running/queued requests,
generation throughput and cache hit rate — but no **prefill or decode speed**,
which is what people look for first. The script appends them:

| Panel | Query |
|---|---|
| Prefill speed, aggregate | `sum(rate(sglang:prompt_tokens_total[$__rate_interval]))` |
| Decode speed, aggregate | `sum(rate(sglang:generation_tokens_total[$__rate_interval]))` |
| Prefill speed per request | avg uncached prompt tokens ÷ avg TTFT |
| Decode speed per request | `1 / avg(inter_token_latency_seconds)` |

Per-request prefill is `uncached_prompt_tokens_histogram_sum / _count` divided by
`time_to_first_token_seconds_sum / _count`. Per-request decode inverts the mean
inter-token latency, which is the same quantity as TPOT.

## Verify before trusting a panel

Query Prometheus directly; an empty result means the panel is broken, not idle:

```bash
curl -s --get --data-urlencode \
  'query=sum(rate(sglang:generation_tokens_total[5m]))' \
  localhost:9090/api/v1/query
```

## Metrics are not the GPU

VRAM, temperature and clocks are `rocm-smi`'s domain and appear in none of these
62 families. A dashboard built only from SGLang metrics cannot show that a card
is throttling, or that weights have spilled to host memory.

## A decode number to compare against

If per-request decode looks wrong, check whether it is context-dependent before
blaming the dashboard: measure a short prompt. On 2x RX 7900 XTX (gfx1100,
TP=2, GPTQ-Int4) decode measured **~129 ms/token regardless of context** — a
5-token prompt and a 171k-token prompt agreed — while GPUs sat at 52% and 83%
busy. That points at launch/all-reduce latency and untuned kernels rather than
bandwidth, and `is_cuda_graph=0` confirms CUDA graphs are off (HIP graph capture
hangs on gfx1100, sglang#30245).

# Qwen3.8-27B on two RX 7900 XTX — corrected deployment guide

Two RX 7900 XTX (gfx1100, 24 GB each, 47.97 GiB combined) serving
**Qwen3.8-27B** with SGLang and the `sglang-radeon-rdna3` plugin. Everything
below was measured on such a host; numbers with no qualification were observed,
not estimated.

This replaces an earlier guide whose two proposed paths were an AWQ setup that
was never run and an FP8 setup that **cannot run on this hardware at all**. The
section [Advice that does not survive contact](#advice-that-does-not-survive-contact)
lists the specific corrections, because the wrong version keeps getting rewritten.

## What fits in 48 GB

The model is 27.32B parameters, dense, and hybrid: 64 layers, of which only 16
are full attention and 48 are linear-attention (Gated DeltaNet), plus an MTP
draft head in the checkpoint.

| Weights | Size | Fits 47.97 GiB? | Keeps MTP head? |
|---|---:|---|---|
| BF16 | 54.6 GB | **no** | yes |
| FP8 | 30.89 GB | yes, in theory | yes |
| AWQ (`mattbucci/Qwen3.8-27B-AWQ`) | 18.73 GB | yes | **no** — 15 files, no `mtp.safetensors` |
| **GPTQ-Int4 (`palmfuture/Qwen3.8-27B-GPTQ-Int4`)** | **21.01 GB** | **yes** | **yes** |

**Use GPTQ-Int4.** It is the only configuration measured end to end here, and the
only quantised one that keeps the MTP head.

### FP8 is not an option, and it is not a flag problem

RDNA3 has no FP8 matrix hardware — FP8 is CDNA3/CDNA4 (gfx942/gfx950). SGLang
declines the FP8 paths on this target, and a `Qwen3.8-27B-FP8` checkpoint loads
and then fails at the first GEMM. Any table row, memory budget or flag list built
around FP8 on gfx1100 describes something that cannot be deployed.

## Install

The prebuilt `sglang-kernel-rdna3` wheel imports
`c10::hip::getCurrentHIPStream`, which **torch 2.9.1 exports and torch 2.13/2.14
do not**. Installing a newer torch and then the prebuilt kernel produces an
`undefined symbol` import failure, four registered architectures and a red
`doctor` — not an obvious ABI message.

```bash
uv venv sglang-env
source sglang-env/bin/activate
export HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTORCH_ROCM_ARCH=gfx1100

# 2.9.1: the version the prebuilt kernel is built against
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/rocm6.4

# installs SGLang + plugin + kernel without letting pip resolve CUDA packages
bash install.sh
```

`install.sh` uses `--no-deps` throughout on purpose: SGLang's core requirements
include `cuda-python`, `flashinfer_python[cu13]` and `sglang-kernel`, and pip will
happily replace the ROCm torch with a CUDA one. The cost is that transitive
requirements are skipped too — `scripts/fill_missing_deps.py` derives those from
installed metadata (a clean venv was missing 69, starting with `urllib3` and
`httpx`).

Verify before serving. For GPTQ this line must not say the ops are absent:

```bash
sglang-rdna3 doctor | grep -i gptq
# [ok  ] gptq ops present (4-bit GPTQ can run)
```

The published image is built from `SGLANG_REF=main`, and upstream deleted the AoT
GPTQ tree after v0.5.19. A kernel built from `main` has no `gptq_*` ops, so the
GPTQ path fails on the first token with
`AttributeError: 'sgl_kernel' object has no attribute 'gptq_shuffle'`. Build the
image with `SGLANG_REF=v0.5.19` if you need GPTQ in a container.

## Serve

```bash
SGLANG_MAMBA_CONV_DTYPE=float16 \
python3 -m sglang.launch_server \
  --model-path ./Qwen3.8-27B-GPTQ \
  --quantization gptq \
  --dtype float16 \
  --served-model-name qwen3.8-27b \
  --tp-size 2 \
  --context-length 196608 \
  --max-running-requests 1 \
  --chunked-prefill-size 2048 \
  --mem-fraction-static 0.80 \
  --max-mamba-cache-size 10 \
  --attention-backend triton \
  --disable-cuda-graph \
  --enable-metrics \
  --host 0.0.0.0 --port 8080
```

Five of these are not optional, and three are non-obvious:

- **`--quantization gptq`** — without it SGLang picks `gptq_marlin`, which is
  CUDA-only on ROCm.
- **`--dtype float16` + `SGLANG_MAMBA_CONV_DTYPE=float16`** — the GDN conv-state
  cache has its own dtype that defaults to bf16 regardless of `--dtype`. Leave it
  and the server starts cleanly, then dies on the first request with
  `Index put requires the source and destination dtypes match`.
- **`--max-mamba-cache-size 10`** — the default 56 slots cost 4.01 GB per card.
  It cannot simply be lowered either: `mamba_ratio=5`, so anything under 5 gives
  `max_num_reqs=0` and the scheduler refuses to start.
- **`--attention-backend triton`** — `fa3`, `flashinfer` and `aiter` are CUDA- or
  CDNA-only.
- **`--disable-cuda-graph`** — HIP graph capture hangs on gfx1100
  (sglang#30245). Omitting it risks a startup hang, not a slowdown.

`--context-length 196608` is 75% of the model's trained 262,144. Prefer a real
window over a round decimal: `128000` wastes what the hardware can hold.

## Budget, measured

At `--context-length 196608`, `--mem-fraction-static 0.80`,
`--max-mamba-cache-size 10`:

| Per card | |
|---|---:|
| GPTQ-Int4 weights | ~10.5 GB |
| KV cache (f16, `max_total_num_tokens=272721`) | 4.16 + 4.16 GB |
| Mamba/SSM state (10 slots) | 0.77 + 0.02 GB |
| **in use** | **~20.9 of 24 GB** |

The pool holds 272,721 tokens against a 196,608 window, so one full-length
session fits with room to spare. Startup: weights 21–33 s, engine ready 56–90 s
(first request compiles Triton kernels and can take longer).

## Validate

```bash
curl -s localhost:8080/health                  # 200 once warm; 503 while compiling
curl -s localhost:8080/v1/models               # id + max_model_len
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"What is 17*23?"}],"max_tokens":600}'
```
Check `max_total_num_tokens` in the startup log against the table above — if it
is below the window, raise `--mem-fraction-static`; if startup OOMs, lower it.
That flag covers weights *plus* KV pool, so the two failure modes point in
opposite directions.

Give reasoning models room: below ~400 `max_tokens` the reply is truncated inside
the thinking block and never reaches an answer.

## Measured performance

Single-stream decode is **~129 ms/token (7.8 tok/s)** — and it is not a
context-length effect: a 5-token prompt and a 171k-token prompt both measure ~130
ms/token. Prefill is healthy (TTFT 0.35 s short, ~11k tok/s on a 171k prompt).

Where the 129 ms goes, measured:

| Measurement | Value |
|---|---|
| All-reduce, 10 KB fp16, 2 ranks | 90.9 µs → ×128/token = **11.6 ms (9%)** |
| Memory bandwidth (stream add, 1 GiB) | 871 GB/s |
| Decode-shaped GEMM (M=1, 5120×17408, bf16) | 0.21 ms at 847 GB/s |
| GPU busy during a 300-token generation | 52% / 83% |

So the cards are not bandwidth-starved and PCIe is not the cause: at 871 GB/s a
token's 10.5 GB of 4-bit weights per card is ~12 ms of work, and they are busy for
~107 ms. The gap is kernel efficiency plus launch overhead — CUDA graphs are
unavailable on gfx1100, and nothing here is autotuned for RDNA3.

**MTP/speculative decoding does not rescue it.** The head is unquantised (BF16/F16)
while the checkpoint's config declares GPTQ: with the inherited `gptq` the draft
accepts nothing (`spec_accept_rate=0`) and decode *worsens* to 157 ms/token, and
`--speculative-draft-model-quantization unquant` makes SGLang derive CUDA-only
`gptq_marlin` and refuse to start.

## Advice that does not survive contact

| Claim | Reality |
|---|---|
| "FP8 path for logic-heavy work" | No FP8 hardware on gfx1100; fails at the first GEMM |
| "`--kv-cache-dtype fp8_e4m3` is non-negotiable for 128k" | KV is 34.8 KB/token; 128k = 4.5 GB. 196,608 fits at f16 with a 272k pool |
| "FP8 128k is at the absolute limits of VRAM" | 30.89 GB weights + 4.5 GB KV = 35.4 of 47.97 GB |
| "`--context-length 128000`" | Arbitrary decimal; trained maximum is 262,144 |
| "OOM → reduce context to 96000" | Lower `--mem-fraction-static` first; that failure is activation headroom |
| "`HIP_FORCE_INT_PEER_TO_PEER=0`" | No such variable in ROCm; `NCCL_P2P_DISABLE=1` is the real lever |
| "`SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION` bounds generation" | Real variable, read only by `srt/disaggregation/decode.py`; inert in a single-node TP=2 launch |
| "`uv pip install sglang sglang-radeon`" | `sglang-radeon` is not on PyPI (it is `sglang-radeon-rdna3`), and plain `sglang` drags in CUDA torch |
| "`--index-url .../nightly/rocm`" | 403; needs a variant such as `nightly/rocm7.2` |
| "`rocm-smi --showmemuse`" | Deprecated; `--showmeminfo vram` is supported |

Monitoring is covered separately in [monitoring.md](monitoring.md): metrics are
exposed as `sglang:xxx` (colon), and the bundled Grafana dashboard queries
`sglang_xxx`, so every panel is blank until corrected.

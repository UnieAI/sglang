# Jacobi Decoding in SGLang

Jacobi Decoding (or Lookahead Decoding) is a parallel decoding algorithm that accelerates inference by predicting and verifying multiple tokens in a single forward pass. Unlike NGRAM, it relies on the model's own capabilities (and optionally an n-gram pool of recent tokens) rather than a strict historical cache match.

## Key Features

- **Self-Speculative**: Uses the target model itself for verification, requiring no external draft model.
- **Prefix-Independent**: Can accelerate generation even without exact historical matches (though benefits from local repetition via the n-gram pool).
- **Dynamic Strategy Control (DSC)**: Automatically enables/disables optimization based on runtime conditions.
- **Fully Compatible**: Works with LMCache and PD Disaggregation.

## Usage

Enable Jacobi Decoding by setting `--speculative-algorithm JACOBI`.

## Behavior and Constraints

- **Greedy-only**: Jacobi currently runs only when all requests in the batch are greedy. If any request uses non-greedy sampling (top-p, top-k, min-p, penalties, logit bias, or `n > 1`), the worker falls back to standard decode for that step.
- **Cache alignment**: Jacobi runs only when `stable_uncached_len == 1` (the normal decode case). If cache misses or retractions make this > 1, Jacobi falls back for that step.
- **Unsupported features**: `return_logprob` and grammar constraints are rejected for Jacobi requests.
- **Model families**: Only causal LLM families allow full logits for Jacobi (Qwen2/Qwen3/Llama/Mistral/Mixtral/Gemma).
- **Step-level fallback**: Constraints and gates are checked per decode step. When any check fails, that step runs as standard decode and Jacobi can resume later.

### Basic Usage

```bash
python -m sglang.launch_server \
  --model <your-model> \
  --speculative-algorithm JACOBI \
  --jacobi-steps-per-yield 5 \
  --jacobi-ngram-pool-size 20
```

## Parameter Guide

Core:
- `--speculative-algorithm JACOBI`: enable Jacobi decoding.
- `--speculative-num-draft-tokens`: block size per step. Total draft length is `block_size * jacobi_num_blocks`.
- `--jacobi-num-blocks`: number of blocks (K). Larger values increase lookahead but cost more compute.
- `--jacobi-steps-per-yield`: refinement steps before yielding to the scheduler. Higher values can improve acceptance but increase per-step latency and reduce fairness.
- `--jacobi-ngram-pool-size`: request-level n-gram pool for tail reuse. Use 0 to disable.
- `--jacobi-prefill-random`: initialize the first draft with random tokens (useful for experimentation; off by default).

Dynamic Strategy Control (DSC) gates:
- `--jacobi-max-batch-size <N>`: disable Jacobi when batch size > N.
- `--jacobi-accept-rate-low <L>`: disable Jacobi if acceptance rate drops below L.
- `--jacobi-accept-rate-high <H>`: re-enable Jacobi when acceptance rises above H.
- `--jacobi-accept-rate-ema-decay <D>`: EMA smoothing for acceptance rate (default 0.9).
- `--jacobi-accept-rate-warmup <W>`: number of steps to warm up before gating.
- `--jacobi-accept-rate-probe-interval <P>`: when disabled, probe every P steps.

Scheduler congestion gate:
- `--jacobi-waiting-running-ratio-high <H>` and `--jacobi-waiting-running-ratio-low <L>`: disable Jacobi when the waiting/running ratio exceeds `H`, re-enable below `L`.

## Performance Notes

- **Low concurrency**: Jacobi is most helpful when acceptance is high and the draft length is moderate. Large drafts or many steps per yield can increase per-step latency and reduce fairness.
- **High concurrency**: When the waiting/running ratio crosses the configured gate, the scheduler disables Jacobi and switches to standard decode, so throughput typically returns to baseline rather than dropping further.
- **Acceptance-driven**: If acceptance is low, Jacobi can be slower than baseline because it computes full logits over the stable + draft window. Use `--jacobi-accept-rate-low` (and optionally a probe interval) to auto-disable in those cases.
- **Cache misses**: Repeated cache misses (`stable_uncached_len > 1`) trigger fallback to standard decode for that step.

## Dynamic Strategy Control (DSC)

To prevent performance degradation in adverse scenarios (e.g., high batch size or low acceptance rates), use the following gating arguments:

- `--jacobi-max-batch-size <N>`: Disable Jacobi when batch size > N.
- `--jacobi-accept-rate-low <L>`: Disable Jacobi if acceptance rate drops below L (e.g. 0.3).
- `--jacobi-accept-rate-high <H>`: Re-enable Jacobi if acceptance rate rises above H (e.g. 0.4).
- `--jacobi-accept-rate-probe-interval <P>`: When disabled, try enabling Jacobi every P steps to check if conditions improved.
- `--jacobi-waiting-running-ratio-high <H>` and `--jacobi-waiting-running-ratio-low <L>`: Disable Jacobi under heavy queue pressure, re-enable when pressure drops.

## Configuration Recipes

### DSC Off (Always Try Jacobi)

This keeps Jacobi enabled unless it must fall back due to constraints (non-greedy sampling or `stable_uncached_len != 1`).

```bash
python -m sglang.launch_server \
  --model <your-model> \
  --speculative-algorithm JACOBI \
  --speculative-num-draft-tokens 8 \
  --jacobi-num-blocks 1 \
  --jacobi-steps-per-yield 1 \
  --jacobi-ngram-pool-size 0
```

### DSC On (Adaptive)

This configuration auto-disables Jacobi when acceptance drops or concurrency rises.

```bash
python -m sglang.launch_server \
  --model <your-model> \
  --speculative-algorithm JACOBI \
  --speculative-num-draft-tokens 8 \
  --jacobi-num-blocks 2 \
  --jacobi-steps-per-yield 1 \
  --jacobi-ngram-pool-size 8 \
  --jacobi-max-batch-size 4 \
  --jacobi-accept-rate-low 0.30 \
  --jacobi-accept-rate-high 0.40 \
  --jacobi-accept-rate-probe-interval 50 \
  --jacobi-waiting-running-ratio-high 2.5 \
  --jacobi-waiting-running-ratio-low 1.8
```

### LMCache & PD Integration

Jacobi Decoding is designed to be orthogonal to memory management:
- **LMCache**: It utilizes the standard ModelRunner, so LMCache's KV offloading and retrieval work seamlessly.
- **PD Separation**: The "Decode" worker handles the Jacobi logic independently. If DSC triggers a fallback, it simply behaves as a standard decode step, maintaining full compatibility with the Prefill-Decode architecture.

## Performance Notes

- **Low concurrency**: When acceptance rates are high, Jacobi can increase tokens per forward and improve throughput. If acceptance is low, extra lookahead can reduce throughput.
- **High concurrency**: The scheduler can disable Jacobi via the waiting/running ratio or max-batch gates. When disabled, the system falls back to standard decode and should approach baseline throughput rather than guaranteeing a slowdown.
- **Tuning for stability**: If you see regressions, lower `--jacobi-num-blocks` or `--jacobi-steps-per-yield`, and tighten the accept-rate or waiting/running gates.

## Comparison: Jacobi vs NGRAM

| Feature | NGRAM | Jacobi |
|---|---|---|
| **Mechanism** | Matches exact history sequences | Model-based lookahead (iteration) |
| **Best For** | Code, Logs, Repetitive Structured Text | General Text, Creative Writing |
| **Throughput** | 5-10x (on hit) | 1.5-3x (more consistent) |
| **Overhead** | Low (CPU lookup) | Medium (Model Forward) |

For most general-purpose serving where user queries vary, Jacobi offers a robust middle ground.

## Theoretical Speedup Analysis

We can model the theoretical speedup $S$ as:

$$ S = \gamma \times \frac{T_{base}}{T_{jacobi}} $$

Where:
- $\gamma$: **Acceptance Rate** (average tokens verified per step). Typically 1.5 ~ 3.5 for Jacobi.
- $T_{base}$: Time per step for standard decoding (1 token).
- $T_{jacobi}$: Time per step for Jacobi decoding (proposing $N$ draft tokens + verification).

### Scenario 1: Low Concurrency (Bandwidth Bound)
In this regime (e.g., small batch size), GPU execution is bound by **Memory Bandwidth** (reading model weights) rather than Compute.
- $T_{jacobi} \approx T_{base}$ (The cost of processing $N$ extra tokens is negligible compared to loading full model weights).
- **Result**: $S \approx \gamma$. You get "free" speedup equal to the acceptance rate (e.g., ~2-3x).

### Scenario 2: High Concurrency (Compute Bound)
In this regime (e.g., large batch size), the GPU is saturated with computation. Processing more tokens costs time linearly.
- Let $N$ be the draft block size (e.g., 8).
- $T_{jacobi} \approx N \times T_{base}$ (roughly proportional to total tokens processed).
- **Result**: $S \approx \gamma \times \frac{1}{N}$.
- Since typically $\gamma < N$ (e.g., accept 3 out of 8), **$S < 1$**. This causes a slowdown.

### Conclusion
This tradeoff explains why **DSC (Dynamic Strategy Control)** and the **Waiting/Running Ratio Gate** are critical. They strictly limit Jacobi to Scenario 1, automatically disabling it when the system moves toward Scenario 2, ensuring you never pay the "Compute Penalty" when the system is already busy.

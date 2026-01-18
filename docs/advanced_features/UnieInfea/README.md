# UnieInfra

[![Website](https://img.shields.io/badge/website-unieai.com-0b7285?style=flat&logo=google-chrome&logoColor=white)](https://unieai.com)
[![Discord](https://img.shields.io/badge/discord-join-5865F2?style=flat&logo=discord&logoColor=white)](https://discord.gg/D4eFuB86bF)
[![GitHub Stars](https://img.shields.io/github/stars/unieai/sglang?style=flat)](https://github.com/unieai/sglang/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/unieai/sglang?style=flat)](https://github.com/unieai/sglang/forks)
[![GitHub Issues](https://img.shields.io/github/issues/unieai/sglang?style=flat)](https://github.com/unieai/sglang/issues)
[![GitHub PRs](https://img.shields.io/github/issues-pr/unieai/sglang?style=flat)](https://github.com/unieai/sglang/pulls)
[![Release](https://img.shields.io/github/v/release/unieai/sglang?style=flat)](https://github.com/unieai/sglang/releases)
[![License: NonCommercial](https://img.shields.io/badge/License-NonCommercial-lightgrey?style=flat)](https://github.com/unieai/sglang/blob/main/LICENSE)
[![CI](https://github.com/unieai/sglang/actions/workflows/pr-test.yml/badge.svg)](https://github.com/unieai/sglang/actions/workflows/pr-test.yml)

English | [中文](README.zh.md)

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Preset Configurations](#preset-configurations)
- [Core Concepts](#core-concepts)
- [Architecture](#architecture)
- [Configuration Guide](#configuration-guide)
- [Usage Examples](#usage-examples)
- [Integration](#integration)
- [Understanding Gates](#understanding-gates)
- [FAQ](#faq)
- [Performance Tips](#performance-tips)
- [Troubleshooting](#troubleshooting)
- [Implementation Details](#implementation-details)
- [Roadmap](#roadmap)
- [Trade-offs](#trade-offs)

## Overview

### What is UnieInfra?

UnieInfra is a **Dynamic Strategy Controller (DSC)** for NGRAM speculative decoding in SGLang. It intelligently decides, at each decode step, whether to use NGRAM speculative decoding or fall back to the standard decode path based on runtime conditions.

### Key Features

- **Per-step decision making**: Evaluates conditions at every decode step without requiring worker restarts
- **Multiple gate mechanisms**: Batch size, sequence length, output length, and accept-rate gating
- **Conservative safety**: Falls back to standard decode when NGRAM would be inefficient
- **Zero-overhead fallback**: Uses existing decode logic and KV cache management
- **Experiment-friendly**: All behavior controlled via CLI parameters
- **PD-compatible**: Works seamlessly with Prefill/Decode disaggregation

### Why Use UnieInfra?

NGRAM speculative decoding can provide significant speedups at low batch sizes and high acceptance rates. However, it can become slower than standard decode when:
- Batch size is large (high concurrency)
- Acceptance rate is low (bad prompt/task match)
- Sequences are very long
- Output requirements are very long

**Without UnieInfra**, NGRAM continues running even when it's inefficient, potentially reducing overall throughput.

**With UnieInfra**, the system automatically switches to standard decode in these scenarios, preserving the benefits of NGRAM where it helps while avoiding performance degradation where it doesn't.

### Latest News

- **2026-01-18**: Complete Documentation Overhaul - Addressing User Configuration Challenges

  **Problem Context**:
  - Original documentation was disorganized with scattered information
  - Parameter descriptions were too brief, users unclear on threshold tuning
  - Lack of practical examples created gap between theory and practice
  - Insufficient troubleshooting guidance made self-service debugging difficult

  **Improvements Delivered**:
  - ✅ **Added "Parameter Quick Reference"**: Comprehensive parameter tables covering SGLang basics, NGRAM parameters, and all DSC gate parameters with descriptions, suggested values, and use cases
  - ✅ **Expanded "Configuration Guide"**: Each gate now has detailed purpose explanation, operation mechanism, usage scenarios, and example configurations
  - ✅ **Added "Preset Configurations"**: Two template configurations - "Optimized Baseline" and "Clean Baseline"
  - ✅ **Added "Usage Examples"**: 5 real-world examples from minimal to full configuration, covering RAG, code generation, chat scenarios
  - ✅ **Added "Understanding Gates"**: Deep dive into why NGRAM can be slower, gate trigger conditions, step-by-step tuning strategy
  - ✅ **Added "Troubleshooting"**: 4 common issues with symptoms, causes, and solutions
  - ✅ **Expanded FAQ**: Increased from 4 to 8 frequently asked questions
  - ✅ **Added "Performance Tips"**: Recommended configurations for different workloads, common pitfalls, monitoring suggestions

  **Expected Impact**:
  - 🎯 New users can complete basic setup and run in 5 minutes
  - 🎯 Production users can select optimal configurations based on workload characteristics
  - 🎯 Reduce performance issues caused by improper parameter settings
  - 🎯 Lower support burden - users can self-resolve 80% of common issues
  - 🎯 English and Chinese documentation fully synchronized with 100% content consistency

## Quick Start

### Minimal Setup

Enable NGRAM with basic batch size gating:

```bash
python3 -m sglang.launch_server \
  --model <hf_model> \
  --speculative-algorithm NGRAM \
  --speculative-ngram-max-batch-size 15
```

### What to Expect

- **Low batch size (≤15 requests)**: NGRAM enabled, faster token generation
- **High batch size (>15 requests)**: Automatic fallback to standard decode
- **Gate status**: Logged in server output (check console for "NGRAM enabled/disabled" messages)

### Verifying It Works

Monitor your server logs for messages indicating gate status changes. You should see the system switch between NGRAM and standard decode based on current batch size.

## Preset Configurations

These presets are **templates**. Fill in your thresholds and model paths as needed.

### Optimized Baseline (Prefill Cache + LMCache + DSC + NGRAM)

Use this when you want a strong default with caching and dynamic switching enabled.

- Prefill cache (RadixAttention prefix cache) is **on by default**. Do not set `--disable-radix-cache`.
- LMCache is enabled explicitly.
- DSC + NGRAM is enabled; you supply thresholds.

```bash
python3 -m sglang.launch_server \
  --model <hf_model> \
  --speculative-algorithm NGRAM \
  --speculative-ngram-max-batch-size <N> \
  --enable-lmcache
```

Optional gates you can add (choose what fits your workload):

```bash
--speculative-ngram-accept-rate-low <L> \
--speculative-ngram-accept-rate-high <H> \
--speculative-ngram-max-seq-len <M> \
--speculative-ngram-max-new-tokens <K>
```

### Clean Baseline (DSC + NGRAM Only)

This is the **minimal execution profile** for UnieInfra: dynamic switching with no extra cache layers.

```bash
python3 -m sglang.launch_server \
  --model <hf_model> \
  --speculative-algorithm NGRAM \
  --speculative-ngram-max-batch-size <N>
```

## Parameter Quick Reference

The following tables list all commonly used parameters for launching UnieInfra.

### Basic Parameters (SGLang)

| Parameter | Required | Description | Example Value |
|-----------|----------|-------------|---------------|
| `--model` | ✅ | HuggingFace model path or local path | `meta-llama/Llama-3.1-8B-Instruct` |
| `--host` | ❌ | Server bind address | `127.0.0.1` (default) or `0.0.0.0` |
| `--port` | ❌ | Server port number | `30000` (default) |
| `--tp-size` | ❌ | Tensor Parallelism size | `1` (default), number of GPUs |
| `--mem-fraction-static` | ❌ | Static GPU memory allocation fraction | `0.9` (default) |

### NGRAM Core Parameters

| Parameter | Required | Description | Example Value |
|-----------|----------|-------------|---------------|
| `--speculative-algorithm` | ✅ | Enable speculative decoding algorithm | `NGRAM` |
| `--speculative-ngram-capacity` | ❌ | NGRAM cache capacity (number of tokens) | `10000000` (default, 10M) |
| `--speculative-ngram-min-match-window-size` | ❌ | Minimum match window size | `1` (default) |
| `--speculative-ngram-max-match-window-size` | ❌ | Maximum match window size | `12` (default) |
| `--speculative-ngram-branch-length` | ❌ | Draft branch length | `18` (default) |

### DSC Gate Parameters (Dynamic Strategy Control)

These parameters control when to enable/disable NGRAM. **All optional** - if not set, that gate won't restrict NGRAM.

| Parameter | Default | Description | Suggested Range |
|-----------|---------|-------------|-----------------|
| `--speculative-ngram-max-batch-size` | None | Use NGRAM only when batch size ≤ N | `10-20` |
| `--speculative-ngram-max-seq-len` | None | Use NGRAM only when sequence length < M | `2048-8192` |
| `--speculative-ngram-max-new-tokens` | None | Use NGRAM only when output length ≤ K | `256-1024` |
| `--speculative-ngram-accept-rate-low` | None | Close gate when EMA < this value | `0.3-0.5` |
| `--speculative-ngram-accept-rate-high` | None | Open gate when EMA ≥ this value | `0.4-0.6` |
| `--speculative-ngram-accept-rate-ema-decay` | `0.9` | EMA decay coefficient (0-1) | `0.85-0.95` |
| `--speculative-ngram-accept-rate-warmup` | `0` | Number of samples before gate applies | `32-128` |
| `--speculative-ngram-accept-rate-probe-interval` | `0` | Probe interval when gate is closed (steps) | `10-50` |

### Optional Feature Parameters

| Parameter | Description | When to Use |
|-----------|-------------|-------------|
| `--enable-lmcache` | Enable hierarchical KV cache | Want larger cache capacity and better hit rates |
| `--disable-radix-cache` | Disable prefix cache | **Not recommended** in most cases |
| `--enable-metrics` | Enable Prometheus metrics | Need monitoring and observability |
| `--log-requests` | Log all request details | Debugging or tracking requests |

### PD Disaggregation Parameters (Advanced)

| Parameter | Description | Example Values |
|-----------|-------------|----------------|
| `--disaggregation-mode` | PD disaggregation mode | `null` (default), `prefill`, `decode` |
| `--disaggregation-transfer-backend` | KV transfer backend | `mooncake`, `nixl`, `ascend` |

### Parameter Usage Examples

#### Minimal Setup (Batch Gate Only)
```bash
python3 -m sglang.launch_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --speculative-algorithm NGRAM \
  --speculative-ngram-max-batch-size 15
```

#### Production Setup (Batch + Accept-Rate Gate)
```bash
python3 -m sglang.launch_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --speculative-algorithm NGRAM \
  --speculative-ngram-max-batch-size 15 \
  --speculative-ngram-accept-rate-low 0.40 \
  --speculative-ngram-accept-rate-high 0.55 \
  --speculative-ngram-accept-rate-warmup 64 \
  --enable-lmcache \
  --enable-metrics
```

#### Full Gate Configuration
```bash
python3 -m sglang.launch_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --speculative-algorithm NGRAM \
  --speculative-ngram-max-batch-size 15 \
  --speculative-ngram-max-seq-len 4096 \
  --speculative-ngram-max-new-tokens 512 \
  --speculative-ngram-accept-rate-low 0.40 \
  --speculative-ngram-accept-rate-high 0.55 \
  --speculative-ngram-accept-rate-ema-decay 0.9 \
  --speculative-ngram-accept-rate-warmup 64 \
  --speculative-ngram-accept-rate-probe-interval 20
```

### Important Notes

> [!IMPORTANT]
> - **Required parameters**: Only `--model` and `--speculative-algorithm NGRAM` are mandatory
> - **All gate parameters are optional**: If you don't set a gate parameter, that gate won't restrict NGRAM
> - **Start simple**: Recommend starting with just `--speculative-ngram-max-batch-size`, observe results, then add more gates
> - **Tune thresholds**: The "suggested ranges" in the table are starting points - adjust based on your actual workload

## Core Concepts

### Prefill vs Decode

- **Prefill**: Compute-intensive phase that processes the full input prompt in parallel
- **Decode**: Memory-intensive phase that generates output tokens one at a time

UnieInfra operates **only during decode**. Prefill behavior is unchanged.

### NGRAM Speculative Decoding

NGRAM spec decoding works by:
1. Looking up previously seen token sequences in an NGRAM cache
2. Using matches to draft multiple candidate tokens
3. Verifying drafts in parallel with the target model
4. Accepting correct tokens and rejecting incorrect ones

**Benefits**: When acceptance rate is high, generates multiple tokens per step
**Costs**: Draft lookup + verification overhead, no overlap between stages currently

### Dynamic Strategy Controller (DSC)

The DSC evaluates multiple "gates" at each decode step to decide whether NGRAM is likely to be beneficial:

```
Current batch → Check gates → Use NGRAM or bypass
```

All gates are **opt-in**. If you don't configure a gate, it doesn't restrict NGRAM usage.

### Gating Philosophy

Gates exist to **prevent performance degradation**, not to limit good scenarios:
- Gates only trigger when NGRAM is likely slower than standard decode
- Multiple gates can be combined for fine-grained control
- Conservative defaults ensure safety over maximum performance

## Architecture

### Component Overview

- **Scheduler**: Orchestrates batch processing, owns DSC state, makes gate decisions
- **ScheduleBatch**: Container holding current requests, sequence lengths, and spec info
- **DSC (within Scheduler)**: Evaluates all gates and determines execution path
- **NGRAM Worker**: Generates drafts, runs verification, manages NGRAM cache, supports bypass
- **Target Worker**: Executes actual model computation (standard decode or verification)
- **Output Processor**: Updates tokens/logprobs, releases KV cache, maintains accept-rate EMA

### Single-Node Data Flow

```
    +----------------------------+
    |        Scheduler Loop      |
    +-------------+--------------+
              |
              v
    +----------------------------+
    |        ScheduleBatch       |
    |    - reqs[]                |
    |    - batch_size()          |
    |    - seq_lens_cpu          |
    +-------------+--------------+
                  |
                  v
    +----------------------------+
    | Dynamic Strategy Controller|
    | (DSC)                      |
    |  gates:                    |
    |   - batch size             |
    |   - accept-rate EMA        |
    |   - max seq len            |
    |   - max_new_tokens         |
    +------+---------------------+
       |                 |
       | use NGRAM       | bypass
       v                 v
+----------------+   +----------------+
| NGRAM Worker   |   | Target Worker  |
| (verify path)  |   | (normal decode)|
+-------+--------+   +--------+-------+
        \                 /
         \               /
          v             v
      +---------------------+
      | Output Processor     |
      | - append output_ids  |
      | - update KV cache    |
      | - update accept EMA  |
      +---------------------+
```

### Decode Step Timeline

```
Step 0: Filter running batch (remove finished/retracted requests)
Step 1: Build decode batch from runnable requests
Step 2: DSC evaluates all gates → decision (NGRAM or bypass)
Step 3: Execute chosen path (NGRAM verify OR standard decode)
Step 4: Process outputs (append tokens, logprobs, release KV)
Step 5: Update accept-rate EMA (NGRAM steps only)
```

### Integration Points

DSC integrates into SGLang at the scheduler level:
- No changes to model executor or attention kernels
- Uses existing KV cache management
- Compatible with all sampling backends
- Works with standard output processing

## Configuration Guide

All DSC parameters are opt-in. Configure only the gates you need.

### Gate 1: Batch Size Control

**Parameter**: `--speculative-ngram-max-batch-size N`

**Purpose**: Limit NGRAM to small batch sizes where it performs best.

**How it works**:
- If unset: Batch size doesn't restrict NGRAM
- If N < 1: Always bypass NGRAM
- If current batch size > N: Bypass to standard decode
- If current batch size ≤ N: Allow NGRAM

**When to use**:
- Your workload has variable concurrency
- You know NGRAM benefits diminish above a certain batch size
- You want the simplest form of dynamic switching

**Example configuration**:
```bash
--speculative-ngram-max-batch-size 15
```

**Typical values**: 10-20 for most workloads, adjust based on your hardware and model

### Gate 2: Sequence Length Limit

**Parameter**: `--speculative-ngram-max-seq-len M`

**Purpose**: Disable NGRAM for long sequences where memory pressure is high.

**How it works**:
- Checks `max(seq_lens_cpu)` in current batch
- If maximum sequence length ≥ M: Bypass to standard decode
- Otherwise: Allow NGRAM (subject to other gates)

**When to use**:
- You serve requests with highly variable context lengths
- Memory bandwidth becomes a bottleneck with long sequences
- NGRAM cache hit rate is low for long contexts

**Example configuration**:
```bash
--speculative-ngram-max-seq-len 4096
```

**Typical values**: 2048-8192 depending on model context window and available memory

### Gate 3: Output Length Limit

**Parameter**: `--speculative-ngram-max-new-tokens K`

**Purpose**: Bypass NGRAM for requests requiring very long outputs.

**How it works**:
- Checks each request's `max_new_tokens` parameter
- If ANY request has max_new_tokens > K: Bypass entire batch
- Otherwise: Allow NGRAM (subject to other gates)

**When to use**:
- You have mixed workloads (some short completions, some long generations)
- Long outputs tend to have lower NGRAM cache hit rates
- You want to prioritize latency for short outputs

**Example configuration**:
```bash
--speculative-ngram-max-new-tokens 512
```

**Typical values**: 256-1024 depending on your typical output distributions

### Gate 4: Accept-Rate EMA with Hysteresis

This is the most sophisticated gate, using runtime statistics to adapt to workload characteristics.

#### Core Parameters

**Low Threshold**: `--speculative-ngram-accept-rate-low L`
- Gate **closes** (bypass NGRAM) when EMA < L
- Typical values: 0.3-0.5

**High Threshold**: `--speculative-ngram-accept-rate-high H`
- Gate **opens** (allow NGRAM) when EMA ≥ H
- Must be ≥ L
- Defaults to L if not set
- Typical values: 0.4-0.6

**EMA Decay**: `--speculative-ngram-accept-rate-ema-decay D`
- Controls how much weight to give recent samples
- D = 0.0: Only latest sample matters
- D = 1.0: History dominates
- Default: 0.9
- Typical values: 0.85-0.95

**Warmup**: `--speculative-ngram-accept-rate-warmup W`
- Number of samples to collect before applying gate
- Prevents premature gating based on insufficient data
- Default: 0 (no warmup)
- Typical values: 32-128

**Probe Interval**: `--speculative-ngram-accept-rate-probe-interval P`
- When gate is closed, allow 1 NGRAM step every P decode steps
- Refreshes EMA to detect if conditions have improved
- P = 0: No probing (gate stays closed until other gates force reopening)
- Typical values: 10-50

#### How Accept-Rate Gating Works

1. **Computation**: After each NGRAM verify step:
   ```
   accept_rate = (accepted_tokens + batch_size) / (batch_size * num_draft_tokens)
   EMA = D * EMA_prev + (1 - D) * accept_rate
   ```

2. **Hysteresis**: Uses two thresholds to avoid oscillation:
   ```
   If EMA < L  → Close gate (start bypassing)
   If EMA ≥ H  → Open gate (allow NGRAM)
   If L ≤ EMA < H → Keep current state
   ```

3. **Warmup**: Gate doesn't apply until W samples collected

4. **Probing**: When closed, periodically allow NGRAM to refresh statistics

#### When to Use Accept-Rate Gating

- **Variable workload quality**: Some prompts get high hit rates, others don't
- **Unknown acceptance patterns**: You don't know in advance which tasks work well
- **Adaptive behavior**: You want the system to learn and adapt automatically

#### Example Configuration

**Conservative** (prioritize stability):
```bash
--speculative-ngram-accept-rate-low 0.45 \
--speculative-ngram-accept-rate-high 0.55 \
--speculative-ngram-accept-rate-ema-decay 0.95 \
--speculative-ngram-accept-rate-warmup 64 \
--speculative-ngram-accept-rate-probe-interval 20
```

**Aggressive** (maximize NGRAM usage):
```bash
--speculative-ngram-accept-rate-low 0.30 \
--speculative-ngram-accept-rate-high 0.35 \
--speculative-ngram-accept-rate-ema-decay 0.85 \
--speculative-ngram-accept-rate-warmup 32 \
--speculative-ngram-accept-rate-probe-interval 50
```

### Parameter Summary Table

| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| `speculative-ngram-max-batch-size` | None | ≥ 0 or None | Limit by batch size |
| `speculative-ngram-max-seq-len` | None | ≥ 1 or None | Limit by max sequence length |
| `speculative-ngram-max-new-tokens` | None | ≥ 0 or None | Limit by max output length |
| `speculative-ngram-accept-rate-low` | None | 0.0-1.0 or None | EMA threshold to close gate |
| `speculative-ngram-accept-rate-high` | None | 0.0-1.0 or None | EMA threshold to open gate |
| `speculative-ngram-accept-rate-ema-decay` | 0.9 | 0.0-1.0 | EMA history weight |
| `speculative-ngram-accept-rate-warmup` | 0 | ≥ 0 | Samples before gating applies |
| `speculative-ngram-accept-rate-probe-interval` | 0 | ≥ 0 | Steps between probes (0=disabled) |

## Usage Examples

### Example 1: Batch-Only Gating (Simplest)

**Use case**: Variable concurrency with known batch size threshold

```bash
python3 -m sglang.launch_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --speculative-algorithm NGRAM \
  --speculative-ngram-max-batch-size 15
```

**Behavior**:
- Batch ≤ 15: NGRAM enabled
- Batch > 15: Standard decode

**Best for**: Simple deployments, predictable workloads

### Example 2: Production Configuration

**Use case**: Production deployment with adaptive behavior

```bash
python3 -m sglang.launch_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --speculative-algorithm NGRAM \
  --speculative-ngram-max-batch-size 15 \
  --speculative-ngram-accept-rate-low 0.40 \
  --speculative-ngram-accept-rate-high 0.55 \
  --speculative-ngram-accept-rate-ema-decay 0.9 \
  --speculative-ngram-accept-rate-warmup 64
```

**Behavior**:
- Combines batch size and accept-rate gating
- Adapts to workload quality automatically
- Warmup prevents premature bypassing

**Best for**: Production systems with mixed workloads

### Example 3: Long Context Configuration

**Use case**: Supporting both short and long context requests

```bash
python3 -m sglang.launch_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --speculative-algorithm NGRAM \
  --speculative-ngram-max-batch-size 15 \
  --speculative-ngram-max-seq-len 4096 \
  --speculative-ngram-max-new-tokens 512
```

**Behavior**:
- NGRAM only for: batch ≤ 15 AND seq_len < 4096 AND output ≤ 512
- Automatically bypasses for long context or long output requests

**Best for**: RAG systems, document QA with variable context lengths

### Example 4: Aggressive Optimization

**Use case**: Maximize NGRAM usage, willing to accept some inefficiency

```bash
python3 -m sglang.launch_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --speculative-algorithm NGRAM \
  --speculative-ngram-max-batch-size 25 \
  --speculative-ngram-accept-rate-low 0.30 \
  --speculative-ngram-accept-rate-high 0.35 \
  --speculative-ngram-accept-rate-ema-decay 0.85 \
  --speculative-ngram-accept-rate-warmup 32 \
  --speculative-ngram-accept-rate-probe-interval 50
```

**Behavior**:
- Higher batch size threshold
- Lower accept-rate requirements
- More aggressive EMA (responds faster to changes)
- More frequent probing

**Best for**: Latency-critical applications, known high-quality workloads

### Example 5: Conservative Configuration

**Use case**: Stability over performance, avoid any risk of slowdown

```bash
python3 -m sglang.launch_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --speculative-algorithm NGRAM \
  --speculative-ngram-max-batch-size 10 \
  --speculative-ngram-accept-rate-low 0.50 \
  --speculative-ngram-accept-rate-high 0.60 \
  --speculative-ngram-accept-rate-ema-decay 0.95 \
  --speculative-ngram-accept-rate-warmup 128 \
  --speculative-ngram-accept-rate-probe-interval 10
```

**Behavior**:
- Strict batch size limit
- High accept-rate requirements
- Slow EMA (more stable, less reactive)
- Conservative probing

**Best for**: SLA-critical deployments, initial testing

## Integration

### PD Disaggregation

UnieInfra is fully compatible with Prefill/Decode disaggregation. In PD mode, **DSC runs only on decode workers**. Prefill workers are unaffected.

#### PD Architecture

```
+-----------+      KV transfer      +-----------+
| Prefill   |  ------------------>  | Decode    |
| Workers   |                       | Workers   |
+-----------+                       +-----+-----+
        \                                 |
         \                                | DSC decides
          \                               | NGRAM/bypass
           \                        +-----+-----+
            \--------------------> | Router (PD) |
                                   +-------------+
```

#### Single-Node PD Setup

```bash
# Prefill worker
python -m sglang.launch_server \
  --model <hf_model> \
  --disaggregation-mode prefill \
  --port 30000

# Decode worker (with UnieInfra)
python -m sglang.launch_server \
  --model <hf_model> \
  --disaggregation-mode decode \
  --speculative-algorithm NGRAM \
  --speculative-ngram-max-batch-size 15 \
  --port 30001

# Router
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://127.0.0.1:30000 \
  --decode http://127.0.0.1:30001 \
  --host 0.0.0.0 \
  --port 8000
```

#### Transfer Backends

SGLang supports multiple KV transfer backends:
- **Mooncake**: High-performance RDMA-based transfer
- **NIXL**: Network-based transfer
- **Ascend**: NPU-specific backend (for Ascend hardware)
- **Fake**: Testing/debugging backend

Specify with `--disaggregation-transfer-backend <backend>`.

For complete PD configuration and multi-node deployment, see `docs/advanced_features/pd_disaggregation.md`.

#### PD-Specific Notes

- DSC decisions apply only to decode workers
- NGRAM cache is maintained on decode workers
- Bypass path still updates NGRAM cache (maintains reversibility)
- Transfer backend choice doesn't affect DSC behavior

### LMCache Integration

LMCache is an optional hierarchical KV cache layer that can improve cache hit rates and reduce memory pressure.

#### What is LMCache?

LMCache adds a secondary cache tier (e.g., CPU memory, SSD) below the GPU KV cache, enabling:
- Larger effective cache capacity
- Better cache hit rates for repetitive queries
- Reduced GPU memory pressure

#### Relationship with UnieInfra

LMCache and UnieInfra are **orthogonal**:
- LMCache affects KV storage and reuse
- UnieInfra affects decode path selection
- They can be used independently or together

#### Enabling LMCache

```bash
python3 -m sglang.launch_server \
  --model <hf_model> \
  --speculative-algorithm NGRAM \
  --speculative-ngram-max-batch-size 15 \
  --enable-lmcache
```

#### Configuration

LMCache doesn't change DSC decisions. Gates still operate based on batch size, sequence length, and accept rates.

For detailed LMCache configuration, see:
- `docs/advanced_features/hicache_design.md`
- `python/sglang/srt/mem_cache/storage/lmcache/README.md`

## Understanding Gates

### Why NGRAM Can Be Slower

NGRAM speculative decoding adds overhead:

1. **Draft lookup**: Searching NGRAM cache for matches
2. **Draft construction**: Building candidate token trees
3. **Verification**: Running verify pass on all candidates
4. **No overlap**: Current implementation doesn't overlap stages

When acceptance is high, accepting multiple tokens per step offsets these costs. When acceptance is low or batch is large:
- Overhead remains constant
- Benefit decreases (fewer accepted tokens)
- Memory bandwidth becomes bottleneck (large batches)

Result: Standard decode can be faster.

### When Gates Trigger: Decision Tree

```
Start decode step
    |
    v
Batch size gate enabled?
    |
    ├─ Yes → Batch > threshold? ─ Yes → BYPASS
    |            |
    |            No
    v            v
Seq len gate enabled?
    |
    ├─ Yes → Max seq ≥ threshold? ─ Yes → BYPASS
    |            |
    |            No
    v            v
Output len gate enabled?
    |
    ├─ Yes → Any output > threshold? ─ Yes → BYPASS
    |            |
    |            No
    v            v
Accept-rate gate enabled?
    |
    ├─ Yes → Warmup complete?
    |            |
    |            ├─ No → (Allow NGRAM, collect stats)
    |            |
    |            v
    |         Gate open?
    |            |
    |            ├─ Yes → (Allow NGRAM)
    |            |
    |            v
    |         EMA < low? ─ Yes → Close gate → BYPASS*
    |            |
    |            No → EMA ≥ high? ─ Yes → Open gate
    |                      |
    |                      No → Keep current state
    v
USE NGRAM (if no bypass triggered)

* With probing: periodically allow NGRAM to refresh EMA
```

### Gate Interactions

Multiple gates work in combination:
- **ALL gates must pass** for NGRAM to run
- **ANY gate triggering** causes bypass
- Gates are evaluated in order (batch size → seq len → output len → accept-rate)
- Early bypass skips later gate evaluations (minor optimization)

### Tuning Strategy

#### Step 1: Start Simple

Begin with batch size gating only:
```bash
--speculative-ngram-max-batch-size 15
```

#### Step 2: Monitor Performance

Watch for:
- Token generation throughput
- Latency (TTFT and TPOT)
- GPU/memory utilization
- NGRAM acceptance rates (if logged)

#### Step 3: Add Accept-Rate Gating

If you see variable quality:
```bash
--speculative-ngram-max-batch-size 15 \
--speculative-ngram-accept-rate-low 0.40 \
--speculative-ngram-accept-rate-high 0.50 \
--speculative-ngram-accept-rate-warmup 64
```

#### Step 4: Tune Thresholds

- **Lower accept-rate thresholds** if NGRAM bypasses too often
- **Higher accept-rate thresholds** if you see slowdowns
- **Adjust batch size** based on your hardware and concurrency patterns

#### Step 5: Add Length Gates (if needed)

If you have long-context workloads:
```bash
--speculative-ngram-max-seq-len 4096 \
--speculative-ngram-max-new-tokens 512
```

## FAQ

### Q: Why is my NGRAM performance worse with DSC?

**A**: DSC is conservative by default. Check if gates are bypassing NGRAM when they shouldn't:
- Lower your accept-rate thresholds
- Increase batch size threshold
- Reduce warmup period
- Check logs for gate status

### Q: How do I know if gates are working correctly?

**A**: Enable request logging and monitor server output:
```bash
--log-level debug
```

Look for messages indicating gate status changes and NGRAM enable/disable events.

### Q: What metrics should I monitor?

**A**: Key metrics:
- **Throughput** (tokens/second): Overall system performance
- **Latency** (TTFT, TPOT): User-facing latency
- **Batch size distribution**: Understanding your workload
- **Accept rate** (if exposed): NGRAM effectiveness
- **Gate trigger frequency**: How often bypasses occur

### Q: How do I debug gate behavior?

**A**: Follow this checklist:
1. Enable debug logging
2. Check current parameter values
3. Monitor batch sizes in real-time
4. Track accept-rate EMA (add logging if needed)
5. Test with synthetic workloads (controlled batch sizes)
6. Compare performance with/without specific gates

### Q: When should I disable a gate?

**A**: Disable a gate when:
- It triggers too often for your workload
- It doesn't provide meaningful protection
- You want to isolate its effect for debugging

Disable by not setting the parameter or setting it to extreme values (e.g., batch size = 1000000).

### Q: Can I change gate parameters without restarting?

**A**: No, gate parameters are set at server startup. You must restart the server to change them.

### Q: Does DSC work with LoRA adapters?

**A**: Yes, DSC operates at the scheduler level and is compatible with LoRA serving.

### Q: What's the performance impact of the DSC itself?

**A**: Minimal. Gate evaluation is simple arithmetic on CPU-side metadata (batch size, sequence lengths, EMA). Overhead is negligible compared to model inference.

## Performance Tips

### Recommended Starting Configurations

**For RAG/Q&A systems**:
```bash
--speculative-ngram-max-batch-size 15 \
--speculative-ngram-max-seq-len 4096 \
--speculative-ngram-accept-rate-low 0.40 \
--speculative-ngram-accept-rate-high 0.55
```

**For code generation**:
```bash
--speculative-ngram-max-batch-size 12 \
--speculative-ngram-max-new-tokens 1024 \
--speculative-ngram-accept-rate-low 0.35 \
--speculative-ngram-accept-rate-high 0.45
```

**For chat/instruction following**:
```bash
--speculative-ngram-max-batch-size 20 \
--speculative-ngram-accept-rate-low 0.45 \
--speculative-ngram-accept-rate-high 0.60
```

### Common Pitfalls

1. **Too aggressive thresholds**: Setting accept-rate thresholds too high disables NGRAM even when beneficial
2. **No warmup**: Premature gating based on insufficient statistics
3. **Batch size too low**: Missing opportunities for NGRAM speedup
4. **Batch size too high**: Allowing NGRAM when standard decode is faster
5. **No probing**: Gate stays closed even when conditions improve

### Monitoring and Observability

Enable metrics for better visibility:
```bash
--enable-metrics \
--log-requests
```

Consider adding custom logging for:
- Current batch sizes
- Gate trigger events
- Accept-rate EMA values
- NGRAM enable/disable transitions

## Troubleshooting

### Issue: NGRAM never triggers

**Symptoms**: Server always uses standard decode

**Possible causes**:
1. Batch size always exceeds threshold
2. Accept-rate gate stuck closed
3. Sequence/output length gates always trigger

**Solutions**:
- Check actual batch sizes in your workload
- Increase batch size threshold
- Lower accept-rate thresholds or disable accept-rate gate
- Review sequence/output length distributions

### Issue: Performance worse than without DSC

**Symptoms**: Lower throughput or higher latency with UnieInfra

**Possible causes**:
1. Gates switching too frequently (high overhead)
2. Thresholds set incorrectly
3. Workload doesn't benefit from NGRAM

**Solutions**:
- Increase EMA decay (more stable)
- Widen hysteresis gap (high - low)
- Disable problematic gates
- Consider if NGRAM is right for your workload

### Issue: Accept-rate gate stays closed

**Symptoms**: Gate closes and never reopens

**Possible causes**:
1. Probe interval = 0 (no probing)
2. EMA never recovers to high threshold
3. Workload quality genuinely poor

**Solutions**:
- Enable probing: `--speculative-ngram-accept-rate-probe-interval 20`
- Lower high threshold
- Check if workload characteristics changed

### Issue: High variability in latency

**Symptoms**: Inconsistent response times

**Possible causes**:
1. Frequent gate transitions
2. Workload has mixed characteristics
3. EMA settings too reactive

**Solutions**:
- Increase EMA decay (smoother transitions)
- Widen hysteresis gap
- Use more conservative thresholds
- Consider separating workload types

## Implementation Details

### Code Locations

Core DSC logic:
- **Gate decision logic**: `python/sglang/srt/managers/scheduler.py`
  - Method: `_use_ngram_for_decode()` (or similar)
  - Evaluates all gates and returns decision

- **Accept-rate EMA update**: `python/sglang/srt/managers/scheduler_output_processor_mixin.py`
  - Updates after each NGRAM verify step

- **NGRAM worker bypass**: `python/sglang/srt/speculative/ngram_worker.py`
  - Implements bypass to standard decode
  - Maintains NGRAM cache even during bypass

- **CLI parameters**: `python/sglang/srt/server_args.py`
  - All `speculative_ngram_*` parameters
  - Validation and default values

### For Contributors

If you're modifying DSC behavior:

1. **Gate logic**: Modify `scheduler.py` gate evaluation
2. **New gates**: Add parameters in `server_args.py`, implement checks in scheduler
3. **EMA calculation**: Adjust in output processor mixin
4. **Testing**: Use controlled synthetic workloads with known batch sizes and accept rates

## Roadmap

Planned integrations for other inference backends:

- [ ] **vLLM**: Implement DSC for vLLM's speculative decoding
- [ ] **TensorRT-LLM**: Support TensorRT decode pipelines
- [ ] **LMDeploy**: Integration with Bytedance LMDeploy
- [ ] **Efficient-Transformers**: Support for Qualcomm AI Cloud

Community contributions welcome!

## Trade-offs

### Advantages

✅ Preserves NGRAM benefits at low batch sizes
✅ Protects throughput at high concurrency
✅ Adapts automatically to workload quality
✅ No worker restart required for switching
✅ Safe fallback using existing decode logic
✅ Fine-grained control via multiple gates

### Limitations

⚠️ Requires tuning for optimal performance
⚠️ Gains depend on workload acceptance patterns
⚠️ Long sequences often bypass NGRAM (limited benefit)
⚠️ Very long outputs typically disable NGRAM
⚠️ Accept-rate gate is reactive, not predictive
⚠️ No built-in metrics dashboard (requires external monitoring)

---

## License

UnieInfra documentation is provided for non-commercial usage. For base code licensing, refer to [LICENSE](https://github.com/unieai/sglang/blob/main/LICENSE).

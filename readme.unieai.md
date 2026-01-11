# Jacobi Forcing Integration for SGLang

This document details the integration of **Jacobi Forcing**—a parallel decoding algorithm—into the **SGLang** inference engine.

## Overview

Jacobi Forcing converts Large Language Models (LLMs) into native causal parallel decoders. By integrating it into SGLang, we leverage SGLang's high-performance infrastructure (RadixAttention, Continuous Batching) to accelerate inference while maintaining generation quality.

## Architecture

The integration is implemented via a new speculative decoding worker: `JacobiWorker`.

- **Class**: `JacobiWorker` (extends or mimics `TpModelWorker`/`EAGLEWorker` interface)
- **Location**: `sglang/srt/speculative/jacobi_worker.py`
- **Mechanism**: 
  - Implements the Jacobi parallel decoding loop (iterative refinement).
  - Uses the `target_worker`'s model runner for batch forward passes.
  - Manages its own "draft" tokens and verification logic within the SGLang batching lifecycle.

## SGLang Feature Support

This integration is designed to be fully compatible with SGLang's advanced features:

### 1. Continuous Batching
**Supported.** 
The `JacobiWorker` operates within the SGLang Scheduler's `ScheduleBatch` mechanism. The Jacobi iterative process (looping for fixed steps, e.g., $K$ steps) occurs within a single `forward_batch_generation` call. This is effectively treated as a "large step" or a "super-step" within the continuous batching handling, ensuring that other requests can be scheduled dynamically around it.

### 2. RadixAttention
**Supported.** 
The integration utilizes SGLang's standard memory pool and cache allocators (`req_to_token_pool`, `token_to_kv_pool`). 
- **Prefix Caching**: The "stable" prefix of the generation is cached normally, allowing for reuse across requests (e.g., system prompts, shared context).
- **Trajectory Management**: The "noisy" future tokens generated during the Jacobi trajectory are managed as ephemeral states. They utilize the KV pool but are carefully managed to avoid polluting the RadixCache with invalid branches.

### 3. Nanoflow Scheduler
**Supported.** 
By adhering to the `SpeculativeAlgorithm` interface and hooking into `sglang.srt.managers.scheduler.Scheduler`, the integration leverages the existing high-performance scheduling logic (closely related to vLLM/Nanoflow designs). This includes:
- Efficient request queue management.
- Dynamic batch formation.
- Optimized kernel execution.

## Usage

To use Jacobi Forcing, launch the SGLang server with the `JACOBI` speculative algorithm:

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-Coder-7B-Instruct \
  --speculative-algorithm JACOBI \
  --port 30000
```

### Key Arguments
- `--speculative-algorithm JACOBI`: Enables the Jacobi Forcing worker.
- `--speculative-num-draft-tokens`: Jacobi block size (draft length) per step.
- `--jacobi-steps-per-yield`: Number of Jacobi refinement steps before yielding to the scheduler.
- `--jacobi-num-blocks`: Number of Jacobi blocks (K) to draft per step.
- `--jacobi-ngram-pool-size`: Request-level n-gram pool size (0 disables).
- `--jacobi-prefill-random`: Use random tokens for prefill draft initialization.

## Implementation Details

- **`sglang/srt/speculative/jacobi_worker.py`**: Core worker implementation.
- **`sglang/srt/speculative/spec_info.py`**: Registration of `JACOBI` enum and worker factory.
- **`sglang/srt/server_args.py`**: CLI arguments.

## Execution Order (Owners/ETA Placeholders)

### Phase 0: Scope and Guardrails
- Owner: @_____ | ETA: ____
- [x] Lock MVP boundaries: batch size = 1, no `return_logprob`, no grammar, no streaming.
- [x] Target model families: Qwen2/Qwen3/Llama/Mistral/Mixtral/Gemma/GPT-OSS (causal LM only).
- [x] Use `--speculative-num-draft-tokens` as Jacobi block size (no new block-size flag for MVP).

### Phase 1: Speculative Algorithm Registration
- Owner: @_____ | ETA: ____
- [x] Register `JACOBI` via `register_speculative_algorithm("JACOBI", worker_cls=JacobiWorker)` in `sglang/srt/speculative/spec_info.py`.
- [x] Ensure no EAGLE flags are set for JACOBI (no draft model).
- [x] Add a startup log line: "JacobiWorker initialized".

### Phase 2: Request State and Prefix Matching
- Owner: @_____ | ETA: ____
- [x] `sglang/srt/managers/schedule_batch.py`: add `Req.jacobi_draft_ids: List[int]` and `Req.jacobi_enabled: bool`.
- [x] `Req.init_next_round_input`: if `jacobi_enabled`, set `fill_ids = origin + output + jacobi_draft_ids`.
- [x] Prefix matching must use only stable tokens: `origin + output`. Do not include `jacobi_draft_ids` in `RadixKey`.

### Phase 3: ScheduleBatch Flow Adjustments
- Owner: @_____ | ETA: ____
- [x] `ScheduleBatch.is_jacobi` as a property: `return self.spec_algorithm == SpeculativeAlgorithm.JACOBI`.
- [x] `prepare_for_decode`: if `is_jacobi`, call `prepare_for_extend()` and force `forward_mode = EXTEND`.
- [x] `new_page_count_next_decode`: for Jacobi, estimate pages using `req.seqlen + len(req.jacobi_draft_ids)` (draft budget), not 1-token decode.

### Phase 4: Result Plumbing
- Owner: @_____ | ETA: ____
- [x] `sglang/srt/managers/utils.py`: add `draft_ids: Optional[torch.Tensor]` to `GenerationBatchResult`.
- [x] `GenerationBatchResult.copy_to_cpu`: copy `draft_ids` to CPU when present.

### Phase 5: JacobiWorker (New)
- Owner: @_____ | ETA: ____
- [x] Add `sglang/srt/speculative/jacobi_worker.py` (no `TpModelWorker` inheritance).
- [x] Accept `ModelWorkerBatch` and build `ForwardBatch` via `ForwardBatch.init_new(...)`.
- [x] Determine `draft_len` from `batch.extend_seq_lens` (batch size = 1).
- [x] Implement Hybrid Yielding: loop `N = --jacobi-steps-per-yield`, default 1.
- [x] Initial draft: if `req.jacobi_draft_ids` is empty, build a first draft from full logits (argmax).
- [x] Verification: compare draft vs greedy logits, accept the longest matching prefix, and generate a new draft for the remaining tail.
- [x] Return `GenerationBatchResult(next_token_ids=accepted_ids, draft_ids=new_draft_ids)`.

### Phase 6: Full Logits (Jacobi Allowlist)
- Owner: @_____ | ETA: ____
- [x] Ensure Jacobi EXTEND always returns full logits via `LogitsProcessor` (not per-model).
- [x] Allowlist causal LLM families (Qwen2/Qwen3/Llama/Mistral/Mixtral/Gemma) for Jacobi.

### Phase 7: Scheduler Output Processing (Jacobi)
- Owner: @_____ | ETA: ____
- [x] Add `process_batch_result_jacobi` in `sglang/srt/managers/scheduler_output_processor_mixin.py`.
- [x] Multi-token acceptance: append each accepted token, check EOS per token, stop on EOS.
- [x] Update `req.jacobi_draft_ids` from `result.draft_ids`.
- [x] KV cache recycling:
  - Compute `new_committed = prefix_len + accepted_len`.
  - Compute `new_allocated = ceil_align(new_committed, page_size)`.
  - Free `out_cache_loc[keep_len:]`, where `keep_len = new_allocated - prefix_len`.
  - Update `req.kv_committed_len = new_committed`, `req.kv_allocated_len = new_allocated`.
  - Update `req_to_token` mapping to shrink to committed length (use `assign_req_to_token_pool_func`).
- [x] Cache stable prefix only: `tree_cache.cache_unfinished_req(req)` when not finished.

### Phase 8: Scheduler Dispatch
- Owner: @_____ | ETA: ____
- [x] In `sglang/srt/managers/scheduler.py`, route Jacobi batches to `process_batch_result_jacobi`.
- [x] Ensure Jacobi requests remain in running batch unless EOS is reached.

### Phase 9: Configuration
- Owner: @_____ | ETA: ____
- [x] Add `--jacobi-steps-per-yield` (default 1) in `sglang/srt/server_args.py`.
- [x] Document new flag under Key Arguments.

### Phase 10: Manual Verification
- Owner: @_____ | ETA: ____
- [ ] Launch:
  ```bash
  python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-Coder-7B-Instruct \
    --speculative-algorithm JACOBI \
    --speculative-num-draft-tokens 64 \
    --jacobi-steps-per-yield 1 \
    --port 30000
  ```
- [ ] Client test:
  ```python
  import sglang as sgl
  sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))
  print(sgl.gen("def fibonacci(n):", max_tokens=64))
  ```
- [ ] Observe logs for "JacobiWorker initialized" and verify no KV cache growth across multiple runs.

---
*Created by Antigravity for UnieAI*

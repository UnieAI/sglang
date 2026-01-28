# Persistent Decode Scheduler (Experimental)

This module provides a GPU task queue and CUDA graph replay path for decode.
It reuses the existing attention kernels (for example Triton) and does not
introduce a new "persistent attention kernel".

## Status

Working minimal path:
- Decoder-only Llama + CUDA Graph + `persistent_triton`
- Prefill, sampling, logits postprocess remain in Python

Not supported yet:
- Speculative decoding
- Two-batch overlap (TBO)
- PDmux
- Pipeline parallel (PP)
- Non-CUDA devices

## How to enable

1. Use the persistent attention backend:
   - `--attention-backend persistent_triton`
2. Explicitly enable the persistent scheduler:
   - `--enable-persistent-gpu-scheduler`
   - or `SGLANG_ENABLE_PERSISTENT_GPU_SCHEDULER=1`
3. Ensure CUDA Graph is enabled (do not pass `--disable-cuda-graph`).

Example:
```
python -m sglang.launch_server \
  --model-path <model> \
  --attention-backend persistent_triton \
  --enable-persistent-gpu-scheduler
```

## What to expect in logs

On startup (rank 0), you should see:
```
Persistent decode scheduler enabled (backend=..., gpu_scheduler=..., validate_steps=...)
```

On first decode that uses the persistent path:
```
Persistent decode path active (gpu_scheduler=..., buckets=[...])
```

## Fallback behavior

If the queue/worker fails (enqueue/drain errors or worker exceptions), the
scheduler disables itself and automatically falls back to normal CUDA graph
replay, so decode does not hang.

## Build requirements

The first use JIT-compiles `task_queue_ops.cu`. You need a working CUDA toolchain,
and the initial compilation can take time.

## Notes

- This is a scheduling feature. The attention kernels are still the same as
  the selected backend (for example Triton).
- If you need a true persistent attention kernel, it must be implemented in
  the attention backend (for example in `triton_ops` or `sgl-kernel`).

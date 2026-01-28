from __future__ import annotations

import bisect
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.persistent.task_queue import (
    BATCH_SIZE,
    BUCKET_ID,
    MODE,
    SEQ_LEN,
    TASK_FIELDS,
    TASK_ID,
    PersistentTaskQueue,
)
from sglang.srt.utils import get_bool_env_var

MODE_DECODE = 1
logger = logging.getLogger(__name__)


@dataclass
class _PendingTask:
    forward_batch: ForwardBatch
    pp_proxy_tensors: Optional[object]
    skip_attn_backend_init: bool
    done_event: threading.Event = field(default_factory=threading.Event)
    output: Optional[object] = None
    error: Optional[BaseException] = None


class PersistentDecodeScheduler:
    """Queue-backed decode scheduler with optional background worker."""

    def __init__(self, model_runner):
        self.model_runner = model_runner
        self.device = (
            torch.device("cuda", model_runner.gpu_id)
            if model_runner.device == "cuda"
            else torch.device(model_runner.device)
        )
        self.queue = PersistentTaskQueue(
            capacity=model_runner.req_to_token_pool.size,
            num_fields=TASK_FIELDS,
            device=self.device,
        )
        self._pending_lock = threading.Lock()
        self._pending_cond = threading.Condition(self._pending_lock)
        self.pending: Dict[int, _PendingTask] = {}
        self.next_task_id = 1
        self.use_gpu_scheduler = bool(
            model_runner.server_args.enable_persistent_gpu_scheduler
            or get_bool_env_var("SGLANG_ENABLE_PERSISTENT_GPU_SCHEDULER", "false")
        )
        self.validate_steps = int(model_runner.server_args.persistent_validate_steps)
        self._worker_thread: Optional[threading.Thread] = None
        self._worker_stop = threading.Event()
        self._graph_runner = None
        self._disabled = False

    def submit_decode(
        self,
        forward_batch: ForwardBatch,
        graph_runner,
        skip_attn_backend_init: bool,
        pp_proxy_tensors: Optional[object],
    ):
        if self.queue is None or self._disabled:
            return self._replay_direct(
                graph_runner,
                forward_batch,
                skip_attn_backend_init=skip_attn_backend_init,
                pp_proxy_tensors=pp_proxy_tensors,
            )
        task_id = self._next_task_id()
        pending_task = _PendingTask(
            forward_batch=forward_batch,
            pp_proxy_tensors=pp_proxy_tensors,
            skip_attn_backend_init=skip_attn_backend_init,
        )
        with self._pending_cond:
            if len(self.pending) >= self.queue.capacity:
                raise RuntimeError("Persistent scheduler queue is full.")
            self.pending[task_id] = pending_task
            self._pending_cond.notify()
        bucket_id = self._select_bucket_id(forward_batch, graph_runner)
        seq_len = (
            int(forward_batch.seq_lens_sum)
            if forward_batch.seq_lens_sum is not None
            else 0
        )
        task = torch.zeros((1, TASK_FIELDS), dtype=torch.int32, device=self.device)
        task[0, TASK_ID] = task_id
        task[0, BATCH_SIZE] = int(forward_batch.batch_size)
        task[0, SEQ_LEN] = seq_len
        task[0, BUCKET_ID] = bucket_id
        task[0, MODE] = MODE_DECODE
        try:
            self.queue.enqueue(task)
        except Exception as exc:
            self._disable_scheduler(f"enqueue failed: {exc}")
            return self._replay_direct(
                graph_runner,
                forward_batch,
                skip_attn_backend_init=skip_attn_backend_init,
                pp_proxy_tensors=pp_proxy_tensors,
            )
        if self.use_gpu_scheduler:
            self._ensure_worker(graph_runner)
            pending_task.done_event.wait()
            if pending_task.error is not None:
                self._disable_scheduler(f"worker failed: {pending_task.error}")
                return self._replay_direct(
                    graph_runner,
                    forward_batch,
                    skip_attn_backend_init=skip_attn_backend_init,
                    pp_proxy_tensors=pp_proxy_tensors,
                )
            output = pending_task.output
        else:
            outputs = self._drain_cpu(graph_runner)
            output = outputs.get(task_id)
            if output is None:
                self._disable_scheduler(
                    "persistent scheduler did not return the task output"
                )
                return self._replay_direct(
                    graph_runner,
                    forward_batch,
                    skip_attn_backend_init=skip_attn_backend_init,
                    pp_proxy_tensors=pp_proxy_tensors,
                )
        return self._validate_and_return(
            output,
            forward_batch,
            skip_attn_backend_init=skip_attn_backend_init,
            pp_proxy_tensors=pp_proxy_tensors,
        )

    def _next_task_id(self) -> int:
        with self._pending_lock:
            task_id = self.next_task_id
            self.next_task_id += 1
        return task_id

    def _ensure_worker(self, graph_runner) -> None:
        if self._disabled:
            return
        if self._worker_thread is not None:
            if self._graph_runner is graph_runner:
                return
            self._graph_runner = graph_runner
            return
        self._graph_runner = graph_runner
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="persistent_decode_scheduler",
            daemon=True,
        )
        self._worker_thread.start()

    def _replay_direct(
        self,
        graph_runner,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool,
        pp_proxy_tensors: Optional[object],
    ):
        output = graph_runner.replay(
            forward_batch,
            skip_attn_backend_init=skip_attn_backend_init,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        return self._validate_and_return(
            output,
            forward_batch,
            skip_attn_backend_init=skip_attn_backend_init,
            pp_proxy_tensors=pp_proxy_tensors,
        )

    def _validate_and_return(
        self,
        output,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool,
        pp_proxy_tensors: Optional[object],
    ):
        if self.validate_steps > 0:
            self._validate_output(
                output,
                forward_batch,
                skip_attn_backend_init=skip_attn_backend_init,
                pp_proxy_tensors=pp_proxy_tensors,
            )
            self.validate_steps -= 1
        return output

    def _worker_loop(self) -> None:
        if self.model_runner.device == "cuda":
            torch.cuda.set_device(self.model_runner.gpu_id)
        while not self._worker_stop.is_set():
            with self._pending_cond:
                while not self.pending and not self._worker_stop.is_set():
                    self._pending_cond.wait(timeout=0.001)
                if self._worker_stop.is_set():
                    break
                pending_count = len(self.pending)
            if pending_count == 0:
                continue
            try:
                tasks, out_count = self._drain_queue(pending_count)
            except Exception as exc:
                self._disable_scheduler(f"drain failed: {exc}")
                self._fail_pending(RuntimeError("Persistent scheduler disabled."))
                continue
            ready = int(out_count.item())
            if ready == 0:
                time.sleep(0.0005)
                continue
            tasks_cpu = tasks[:ready].cpu()
            for row in tasks_cpu:
                task_id = int(row[TASK_ID].item())
                with self._pending_lock:
                    pending = self.pending.pop(task_id, None)
                if pending is None:
                    continue
                bucket_id = int(row[BUCKET_ID].item())
                if bucket_id > 0:
                    pending.forward_batch.cuda_graph_bucket = bucket_id
                try:
                    output = self._graph_runner.replay(
                        pending.forward_batch,
                        skip_attn_backend_init=pending.skip_attn_backend_init,
                        pp_proxy_tensors=pending.pp_proxy_tensors,
                    )
                    pending.output = output
                except BaseException as exc:
                    pending.error = exc
                finally:
                    pending.forward_batch.cuda_graph_bucket = None
                pending.done_event.set()

    def _drain_cpu(self, graph_runner):
        with self._pending_lock:
            pending_count = len(self.pending)
        if pending_count == 0:
            return {}
        tasks, out_count = self.queue.drain(pending_count)
        ready = int(out_count.item())
        if ready == 0:
            return {}
        tasks_cpu = tasks[:ready].cpu()
        outputs = {}
        for row in tasks_cpu:
            task_id = int(row[TASK_ID].item())
            with self._pending_lock:
                pending = self.pending.pop(task_id, None)
            if pending is None:
                continue
            output = graph_runner.replay(
                pending.forward_batch,
                skip_attn_backend_init=pending.skip_attn_backend_init,
                pp_proxy_tensors=pending.pp_proxy_tensors,
            )
            outputs[task_id] = output
        return outputs

    def _drain_queue(self, pending_count: int):
        if pending_count <= 0:
            empty = torch.empty((0, TASK_FIELDS), dtype=torch.int32, device=self.device)
            out_count = torch.zeros((1,), dtype=torch.int32, device=self.device)
            return empty, out_count
        if self.queue is None:
            raise RuntimeError("Persistent scheduler queue unavailable.")
        bucket_sizes = None
        if (
            self.model_runner.persistent_backend is not None
            and getattr(self.model_runner.persistent_backend, "decode_bucket_sizes_gpu", None)
            is not None
        ):
            bucket_sizes = self.model_runner.persistent_backend.decode_bucket_sizes_gpu
        tasks, out_count = self.queue.drain_with_bucket(
            pending_count,
            bucket_sizes=bucket_sizes,
            bucket_field=BUCKET_ID,
            select_field=BATCH_SIZE,
        )
        return tasks, out_count

    def _fail_pending(self, exc: BaseException) -> None:
        with self._pending_lock:
            pending = list(self.pending.values())
            self.pending.clear()
        for task in pending:
            task.error = exc
            task.done_event.set()

    def _disable_scheduler(self, reason: str) -> None:
        if self._disabled:
            return
        logger.warning("Disabling persistent scheduler: %s", reason)
        self._disabled = True
        self.use_gpu_scheduler = False
        self._worker_stop.set()
        self.queue = None
        self._fail_pending(RuntimeError("Persistent scheduler disabled."))

    def _select_bucket_id(self, forward_batch: ForwardBatch, graph_runner) -> int:
        backend = getattr(self.model_runner, "persistent_backend", None)
        buckets = []
        if backend is not None and backend.decode_graph_cache is not None:
            buckets = backend.decode_graph_cache.buckets()
        if not buckets:
            buckets = list(getattr(graph_runner, "capture_bs", []))
        if not buckets:
            return -1
        raw_bs = int(forward_batch.batch_size)
        idx = bisect.bisect_left(buckets, raw_bs)
        if idx >= len(buckets):
            idx = len(buckets) - 1
        return int(buckets[idx])

    def _validate_output(
        self,
        output,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool,
        pp_proxy_tensors: Optional[object],
    ) -> None:
        if self.model_runner.token_to_kv_pool_allocator is None:
            self._validate_logits_only(output, forward_batch)
            return
        kv_indices = forward_batch.out_cache_loc
        if kv_indices is None or kv_indices.numel() == 0:
            self._validate_logits_only(output, forward_batch)
            return
        allocator = self.model_runner.token_to_kv_pool_allocator
        kv_cache_cpu = allocator.get_cpu_copy(kv_indices)
        try:
            reference = self.model_runner.forward_decode(
                forward_batch,
                skip_attn_backend_init=skip_attn_backend_init,
                pp_proxy_tensors=pp_proxy_tensors,
            )
        finally:
            allocator.load_cpu_copy(kv_cache_cpu, kv_indices)
        self._compare_outputs(output, reference, forward_batch)
        self._validate_logits_only(output, forward_batch)

    def _compare_outputs(self, output, reference, forward_batch: ForwardBatch) -> None:
        if isinstance(output, tuple) and isinstance(reference, tuple):
            for out_item, ref_item in zip(output, reference):
                self._compare_outputs(out_item, ref_item, forward_batch)
            return
        if not isinstance(output, LogitsProcessorOutput) or not isinstance(
            reference, LogitsProcessorOutput
        ):
            return
        logits = self._select_logits(output)
        ref_logits = self._select_logits(reference)
        if logits is None or ref_logits is None:
            return
        if logits.shape != ref_logits.shape:
            raise RuntimeError(
                "Persistent decode validation shape mismatch: "
                f"{logits.shape} vs {ref_logits.shape}"
            )
        diff = (logits - ref_logits).abs().max().item()
        if not torch.allclose(logits, ref_logits, rtol=1e-3, atol=1e-3):
            raise RuntimeError(
                "Persistent decode validation mismatch (max abs diff "
                f"{diff:.6f})."
            )

    @staticmethod
    def _select_logits(output: LogitsProcessorOutput) -> Optional[torch.Tensor]:
        if output.next_token_logits is not None:
            return output.next_token_logits
        return output.full_logits

    def _validate_logits_only(self, output, forward_batch: ForwardBatch) -> None:
        if isinstance(output, tuple):
            for item in output:
                self._validate_logits_only(item, forward_batch)
            return
        if not isinstance(output, LogitsProcessorOutput):
            return
        logits = output.next_token_logits
        if logits is None:
            logits = output.full_logits
        if logits is None:
            return
        expected = len(forward_batch.input_ids)
        if logits.shape[0] != expected:
            raise RuntimeError(
                f"Persistent decode logits shape mismatch: {logits.shape[0]} vs {expected}"
            )
        if not torch.isfinite(logits).all().item():
            raise RuntimeError("Persistent decode logits contain NaN/Inf.")

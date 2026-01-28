from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import torch
from torch.utils.cpp_extension import load

logger = logging.getLogger(__name__)

_ABS_PATH = os.path.dirname(os.path.abspath(__file__))
_MODULE = None
_LOAD_LOGGED = False

TASK_FIELDS = 8
TASK_ID = 0
BATCH_SIZE = 1
SEQ_LEN = 2
BUCKET_ID = 3
MODE = 4


def _load_module():
    global _MODULE
    global _LOAD_LOGGED
    if _MODULE is None:
        if not torch.cuda.is_available():
            raise RuntimeError("Persistent task queue requires CUDA.")
        if not _LOAD_LOGGED:
            logger.info(
                "JIT compiling persistent task queue extension (task_queue_ops.cu); "
                "this may take a while and requires a working CUDA toolchain."
            )
            _LOAD_LOGGED = True
        _MODULE = load(
            name="sglang_persistent_task_queue",
            sources=[os.path.join(_ABS_PATH, "task_queue_ops.cu")],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    return _MODULE


class PersistentTaskQueue:
    """GPU task queue skeleton for the persistent backend."""

    def __init__(
        self,
        capacity: int,
        num_fields: int = TASK_FIELDS,
        device: Optional[torch.device] = None,
    ):
        if num_fields <= 0:
            raise ValueError("num_fields must be positive.")
        self.capacity = int(capacity)
        self.num_fields = int(num_fields)
        self.device = device or torch.device("cuda")
        self.queue = torch.empty(
            (self.capacity, self.num_fields),
            dtype=torch.int32,
            device=self.device,
        )
        self.head = torch.zeros((1,), dtype=torch.int32, device=self.device)
        self.tail = torch.zeros((1,), dtype=torch.int32, device=self.device)
        self.count = torch.zeros((1,), dtype=torch.int32, device=self.device)

    def reset(self) -> None:
        self.head.zero_()
        self.tail.zero_()
        self.count.zero_()

    def enqueue(self, tasks: torch.Tensor) -> torch.Tensor:
        if tasks.numel() == 0:
            return torch.empty((0,), dtype=torch.int32, device=self.device)
        if tasks.dim() != 2 or tasks.shape[1] != self.num_fields:
            raise ValueError(
                f"tasks must have shape [N, {self.num_fields}], got {tuple(tasks.shape)}"
            )
        if tasks.device != self.device:
            tasks = tasks.to(self.device)
        tasks = tasks.to(dtype=torch.int32).contiguous()
        status = torch.empty(
            (tasks.shape[0],), dtype=torch.int32, device=self.device
        )
        _load_module().enqueue(
            tasks,
            self.queue,
            self.tail,
            self.count,
            self.capacity,
            status,
        )
        return status

    def dequeue(self) -> Tuple[torch.Tensor, torch.Tensor]:
        out = torch.empty((self.num_fields,), dtype=torch.int32, device=self.device)
        status = torch.empty((1,), dtype=torch.int32, device=self.device)
        _load_module().dequeue(
            self.queue,
            self.head,
            self.count,
            self.capacity,
            out,
            status,
        )
        return out, status

    def drain(self, max_tasks: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if max_tasks <= 0:
            raise ValueError("max_tasks must be positive.")
        out = torch.empty(
            (max_tasks, self.num_fields), dtype=torch.int32, device=self.device
        )
        out_count = torch.empty((1,), dtype=torch.int32, device=self.device)
        _load_module().drain(
            self.queue,
            self.head,
            self.count,
            self.capacity,
            out,
            out_count,
        )
        return out, out_count

    def drain_with_bucket(
        self,
        max_tasks: int,
        bucket_sizes: Optional[torch.Tensor],
        bucket_field: int,
        select_field: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if max_tasks <= 0:
            raise ValueError("max_tasks must be positive.")
        out = torch.empty(
            (max_tasks, self.num_fields), dtype=torch.int32, device=self.device
        )
        out_count = torch.empty((1,), dtype=torch.int32, device=self.device)
        bucket_sizes = (
            bucket_sizes.to(device=self.device, dtype=torch.int32).contiguous()
            if bucket_sizes is not None and bucket_sizes.numel() > 0
            else None
        )
        _load_module().drain_with_bucket(
            self.queue,
            self.head,
            self.count,
            self.capacity,
            out,
            out_count,
            bucket_sizes,
            int(bucket_field),
            int(select_field),
        )
        return out, out_count

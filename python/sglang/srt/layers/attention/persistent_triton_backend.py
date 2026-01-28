from __future__ import annotations

import bisect
from typing import Any, Iterable, List, Optional

import torch

from sglang.srt.layers.attention.triton_backend import TritonAttnBackend


class _GraphBucketCache:
    """Lightweight bucket cache scaffold for persistent backend graphs."""

    def __init__(self, buckets: Optional[Iterable[int]] = None):
        self._buckets: List[int] = []
        self._cache: dict[int, Any] = {}
        self.set_buckets(buckets)

    def set_buckets(self, buckets: Optional[Iterable[int]]) -> None:
        self._buckets = sorted(set(int(b) for b in buckets)) if buckets else []
        self._cache.clear()

    def select_bucket(self, size: int) -> Optional[int]:
        if not self._buckets:
            return None
        idx = bisect.bisect_left(self._buckets, size)
        if idx >= len(self._buckets):
            idx = len(self._buckets) - 1
        return self._buckets[idx]

    def get(self, size: int) -> Optional[Any]:
        bucket = self.select_bucket(size)
        if bucket is None:
            return None
        return self._cache.get(bucket)

    def put(self, size: int, value: Any) -> None:
        bucket = self.select_bucket(size)
        if bucket is None:
            return
        self._cache[bucket] = value

    def buckets(self) -> List[int]:
        return list(self._buckets)


class PersistentTritonAttnBackend(TritonAttnBackend):
    """Persistent backend scaffold that mirrors Triton and adds graph buckets."""

    def __init__(self, model_runner, skip_prefill: bool = False, kv_indptr_buf=None):
        super().__init__(
            model_runner,
            skip_prefill=skip_prefill,
            kv_indptr_buf=kv_indptr_buf,
        )
        # Marker for higher-level components to enable persistent scheduling.
        self.is_persistent_backend = True
        server_args = model_runner.server_args
        decode_buckets = None
        if not server_args.disable_cuda_graph:
            decode_buckets = server_args.cuda_graph_bs
        prefill_buckets = None
        if server_args.enable_piecewise_cuda_graph:
            prefill_buckets = server_args.piecewise_cuda_graph_tokens

        self.decode_graph_cache = _GraphBucketCache(decode_buckets)
        self.prefill_graph_cache = _GraphBucketCache(prefill_buckets)
        self.decode_bucket_sizes_gpu = None
        self.prefill_bucket_sizes_gpu = None

    def set_decode_graph_buckets(self, buckets: Optional[Iterable[int]]) -> None:
        self.decode_graph_cache.set_buckets(buckets)
        self.decode_bucket_sizes_gpu = self._buckets_to_tensor(
            self.decode_graph_cache.buckets()
        )

    def set_prefill_graph_buckets(self, buckets: Optional[Iterable[int]]) -> None:
        self.prefill_graph_cache.set_buckets(buckets)
        self.prefill_bucket_sizes_gpu = self._buckets_to_tensor(
            self.prefill_graph_cache.buckets()
        )

    def get_decode_graph(self, batch_size: int):
        return self.decode_graph_cache.get(batch_size)

    def put_decode_graph(self, batch_size: int, graph: Any) -> None:
        self.decode_graph_cache.put(batch_size, graph)

    def get_prefill_graph(self, num_tokens: int):
        return self.prefill_graph_cache.get(num_tokens)

    def put_prefill_graph(self, num_tokens: int, graph: Any) -> None:
        self.prefill_graph_cache.put(num_tokens, graph)

    def _buckets_to_tensor(self, buckets: List[int]) -> Optional[torch.Tensor]:
        if not buckets:
            return None
        return torch.tensor(buckets, dtype=torch.int32, device=self.device)

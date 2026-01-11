import logging
from typing import Optional

import torch

from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class JacobiWorker:
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.target_worker = target_worker
        self.model_config = target_worker.model_config
        self.model_runner = target_worker.model_runner
        self.device = self.model_runner.device
        self.block_size = server_args.speculative_num_draft_tokens or 1
        self.num_blocks = max(server_args.jacobi_num_blocks, 1)
        self.steps_per_yield = server_args.jacobi_steps_per_yield
        self.ngram_pool_size = max(server_args.jacobi_ngram_pool_size, 0)
        self.prefill_random = server_args.jacobi_prefill_random
        self.vocab_size = self.model_runner.model_config.vocab_size

        logger.info("JacobiWorker initialized")

    def _append_ngram_pool(self, req, tokens):
        if self.ngram_pool_size <= 0:
            return
        pool = req.jacobi_ngram_pool
        pool.append(tokens)
        if len(pool) > self.ngram_pool_size:
            del pool[0]

    def _select_ngram_tail(self, req, length):
        if self.ngram_pool_size <= 0 or length <= 0:
            return None
        pool = req.jacobi_ngram_pool
        if not pool:
            return None
        candidate = pool[-1]
        if not candidate:
            return None
        if len(candidate) >= length:
            return candidate[:length]
        pad = [candidate[-1]] * (length - len(candidate))
        return candidate + pad

    def forward_batch_generation(
        self,
        model_worker_batch,
        forward_batch: Optional[ForwardBatch] = None,
        **_,
    ) -> GenerationBatchResult:
        if forward_batch is None:
            forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)

        if not model_worker_batch.reqs or len(model_worker_batch.reqs) != 1:
            raise ValueError("Jacobi MVP expects batch size = 1.")

        req = model_worker_batch.reqs[0]
        extend_len = (
            model_worker_batch.extend_seq_lens[0]
            if model_worker_batch.extend_seq_lens
            else forward_batch.input_ids.numel()
        )
        prefix_len = len(req.prefix_indices)
        stable_len = len(req.origin_input_ids) + len(req.output_ids)
        stable_uncached_len = max(stable_len - prefix_len, 0)
        draft_len = extend_len - stable_uncached_len
        if draft_len <= 0:
            draft_len = len(req.jacobi_draft_ids) or (self.block_size * self.num_blocks)

        stable_prefix = None
        if stable_uncached_len > 0:
            stable_prefix = forward_batch.input_ids[:stable_uncached_len].clone()

        draft_ids = req.jacobi_draft_ids
        if not draft_ids:
            if self.prefill_random and self.vocab_size > 0:
                draft_ids = torch.randint(
                    0, self.vocab_size, (draft_len,), device=self.device
                ).tolist()
            else:
                last_token = (
                    req.output_ids[-1] if req.output_ids else req.origin_input_ids[-1]
                )
                draft_ids = [last_token] * draft_len
        elif len(draft_ids) != draft_len:
            if len(draft_ids) > draft_len:
                draft_ids = draft_ids[:draft_len]
            else:
                last_token = draft_ids[-1]
                draft_ids = draft_ids + [last_token] * (draft_len - len(draft_ids))

        draft_tensor = torch.tensor(draft_ids, dtype=torch.long, device=self.device)
        accepted_ids = torch.empty((0,), dtype=torch.long, device=self.device)
        new_draft = draft_tensor
        bootstrap = req.jacobi_needs_bootstrap
        can_run_cuda_graph = False
        logits_output = None

        steps = max(self.steps_per_yield, 1)
        for _ in range(steps):
            if stable_prefix is None:
                input_ids = new_draft
            else:
                input_ids = torch.cat([stable_prefix, new_draft], dim=0)

            if forward_batch.input_ids.numel() == input_ids.numel():
                forward_batch.input_ids.copy_(input_ids)
            else:
                forward_batch.input_ids = input_ids

            logits_output, can_run_cuda_graph = self.model_runner.forward(
                forward_batch, pp_proxy_tensors=None
            )

            logits = logits_output.full_logits
            if logits is None:
                raise RuntimeError("Jacobi requires full_logits but got None.")
            if logits.dim() == 3:
                logits = logits[0]

            draft_logits = logits[-draft_len:] if draft_len > 0 else logits
            if bootstrap:
                new_draft = torch.argmax(draft_logits, dim=-1)
                accepted_ids = torch.empty((0,), dtype=torch.long, device=self.device)
                bootstrap = False
                continue

            current_draft = new_draft
            had_rejection = False
            if new_draft.numel() < 2:
                accepted_ids = new_draft
                next_token = torch.argmax(draft_logits[-1], dim=-1)
                new_draft = next_token.view(1)
            else:
                greedy_tokens = torch.argmax(draft_logits[:-1], dim=-1)
                mismatch = current_draft[1:] != greedy_tokens
                accepted_len = int((mismatch.cumsum(0) == 0).sum().item()) + 1
                accepted_ids = current_draft[:accepted_len]
                had_rejection = accepted_len < current_draft.numel()

                next_token = torch.argmax(draft_logits[accepted_len - 1], dim=-1)
                if accepted_len < current_draft.numel():
                    tail_logits = draft_logits[accepted_len:-1]
                    if tail_logits.numel() > 0:
                        greedy_tail = torch.argmax(tail_logits, dim=-1)
                        new_draft = torch.cat([next_token.view(1), greedy_tail], dim=0)
                    else:
                        new_draft = next_token.view(1)
                else:
                    new_draft = next_token.view(1)

            pool_tail = self._select_ngram_tail(req, draft_len - 1)
            if had_rejection and pool_tail is not None:
                new_draft = torch.tensor(
                    [int(next_token.item())] + pool_tail,
                    dtype=torch.long,
                    device=self.device,
                )

            if new_draft.numel() < draft_len:
                pad_len = draft_len - new_draft.numel()
                pad_token = new_draft[-1]
                pad_tokens = pad_token.repeat(pad_len)
                new_draft = torch.cat([new_draft, pad_tokens], dim=0)
            elif new_draft.numel() > draft_len:
                new_draft = new_draft[:draft_len]

            if self.ngram_pool_size > 0:
                self._append_ngram_pool(req, new_draft.tolist())

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=accepted_ids,
            draft_ids=new_draft,
            can_run_cuda_graph=can_run_cuda_graph,
        )

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import triton

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import apply_custom_logit_processor
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool
from sglang.srt.utils import next_power_of_2

logger = logging.getLogger(__name__)


@dataclass
class LookaheadVerifyInput(SpecInput):
    def __init__(
        self,
        draft_token: torch.Tensor,
        custom_mask: Optional[torch.Tensor],
        positions: torch.Tensor,
        draft_token_num: int,
        lookahead_len: int,
        debug: bool = False,
        keep_prefix_last_token: bool = False,
        apply_prefix_drop: bool = False,
        prefix_only_mask: bool = False,
        use_custom_mask: bool = True,
    ):
        super().__init__(SpecInputType.LOOKAHEAD_VERIFY)
        self.draft_token = draft_token
        self.custom_mask = custom_mask
        self.positions = positions
        self.draft_token_num = draft_token_num
        self.lookahead_len = lookahead_len
        if self.draft_token is not None:
            self.device = self.draft_token.device
        else:
            self.device = self.custom_mask.device
        self.root_len = 1
        self.debug = debug
        self.keep_prefix_last_token = keep_prefix_last_token
        self.apply_prefix_drop = apply_prefix_drop
        self.prefix_only_mask = prefix_only_mask
        self.use_custom_mask = use_custom_mask
        self.drop_prefix_last_token = (
            apply_prefix_drop
            and (not keep_prefix_last_token)
            and (not prefix_only_mask)
        )
        # For ForwardBatch sizing in target_verify.
        self.num_tokens_per_batch = draft_token_num
        self.num_tokens_for_logprob_per_batch = draft_token_num

        self.accepted_indices: Optional[torch.Tensor] = None
        self.accept_length: Optional[torch.Tensor] = None
        self.predict: Optional[torch.Tensor] = None
        self.verified_id: Optional[torch.Tensor] = None
        self.predicted_pairs: Optional[List[List[Tuple[int, int]]]] = None
        self.accept_last_token: Optional[torch.Tensor] = None
        self.accept_last_token_cpu: Optional[torch.Tensor] = None
        self.mismatch_token: Optional[torch.Tensor] = None
        self.mismatch_token_cpu: Optional[torch.Tensor] = None

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return self.draft_token_num, self.draft_token_num

    def prepare_for_verify(self, batch: ScheduleBatch, page_size: int):
        if batch.forward_mode.is_idle():
            return

        batch.input_ids = self.draft_token

        if page_size == 1:
            batch.out_cache_loc = alloc_token_slots(
                batch.tree_cache, len(batch.input_ids)
            )
            end_offset = batch.seq_lens + self.draft_token_num
        else:
            prefix_lens = batch.seq_lens
            prefix_lens_cpu = batch.seq_lens_cpu
            end_offset = prefix_lens + self.draft_token_num
            end_offset_cpu = prefix_lens_cpu + self.draft_token_num
            last_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token,
                batch.req_pool_indices,
                prefix_lens,
            )
            batch.out_cache_loc = alloc_paged_token_slots_extend(
                batch.tree_cache,
                prefix_lens,
                prefix_lens_cpu,
                end_offset,
                end_offset_cpu,
                last_loc,
                len(batch.input_ids),
            )
            self.last_loc = last_loc

        bs = batch.batch_size()
        assign_req_to_token_pool[(bs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            end_offset,
            batch.out_cache_loc,
            batch.req_to_token_pool.req_to_token.shape[1],
            triton.next_power_of_2(bs),
        )


        if self.debug:
            draft_num = self.draft_token_num
            req_to_token = batch.req_to_token_pool.req_to_token
            accept_counts_cpu = (self.accept_length.cpu() + 1).clamp(min=0)
            offsets_cpu = torch.cumsum(accept_counts_cpu, dim=0) - accept_counts_cpu
            for i, req in enumerate(batch.reqs):
                seq_len = int(batch.seq_lens[i].item())
                pool_idx = int(batch.req_pool_indices[i].item())
                out_start = i * draft_num
                out_end = out_start + draft_num
                out_loc = batch.out_cache_loc[out_start:out_end].tolist()
                pool_slice = req_to_token[
                    pool_idx, seq_len : seq_len + draft_num
                ].tolist()
                prefix_tail = req_to_token[
                    pool_idx, max(seq_len - 8, 0) : seq_len
                ].tolist()
                mismatch = [
                    j
                    for j, (a, b) in enumerate(zip(out_loc, pool_slice))
                    if a != b
                ]
                # logger.info(
                #     "lookahead kvmap rid=%s pool_idx=%d seq_len=%d out_loc=%s "
                #     "pool_slice=%s prefix_tail_kv=%s mismatch=%s",
                #     req.rid,
                #     pool_idx,
                #     seq_len,
                #     out_loc,
                #     pool_slice,
                #     prefix_tail,
                #     mismatch,
                # )

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
    ):
        bs = len(req_pool_indices)

        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device=self.device)
        paged_kernel_lens = paged_kernel_lens + self.draft_token_num
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        self.qo_indptr = (
            torch.arange(0, bs + 1, dtype=torch.int32, device=self.device)
            * self.draft_token_num
        )

        kv_indices = torch.empty(
            cum_kv_seq_len[-1], dtype=torch.int32, device=self.device
        )

        from sglang.srt.layers.attention.utils import (
            create_flashinfer_kv_indices_triton,
        )

        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.size(1),
        )
        return kv_indices, cum_kv_seq_len, self.qo_indptr, self.custom_mask

    def _fill_requests(
        self,
        batch: ScheduleBatch,
        accept_counts: torch.Tensor,
        accepted_tokens: torch.Tensor,
    ) -> torch.Tensor:
        accept_counts_cpu = accept_counts.cpu()
        accepted_tokens_cpu = accepted_tokens.cpu()
        offsets = torch.cumsum(accept_counts_cpu, dim=0) - accept_counts_cpu
        adjusted_counts = torch.zeros_like(accept_counts_cpu)

        for i, req in enumerate(batch.reqs):
            count = int(accept_counts_cpu[i].item())
            if count <= 0:
                req.spec_verify_ct += 1
                continue
            start = int(offsets[i].item())
            end = start + count
            token_list = accepted_tokens_cpu[start:end].tolist()
            actual = 0

            for token_id in token_list:
                token_id = int(token_id)
                req.output_ids.append(token_id)
                req.check_finished()
                actual += 1
                if req.finished():
                    break
                if req.grammar is not None:
                    try:
                        req.grammar.accept_token(token_id)
                    except ValueError as e:
                        logger.info(f"{i=}, {req=}")
                        raise e

            req.spec_verify_ct += 1
            req.spec_accepted_tokens += actual
            adjusted_counts[i] = actual

        return adjusted_counts

    def _free_cache(
        self, batch: ScheduleBatch, page_size: int, accept_length_cpu: torch.Tensor
    ):
        bs = batch.batch_size()
        if page_size == 1:
            evict_mask = torch.full_like(self.draft_token, True, dtype=torch.bool)
            evict_mask[self.accepted_indices] = False
            batch.token_to_kv_pool_allocator.free(batch.out_cache_loc[evict_mask])
            batch.out_cache_loc = batch.out_cache_loc[self.accepted_indices]
        else:
            # Keep accepted tokens in-place to avoid root-shift issues, and free others.
            evict_mask = torch.full_like(self.draft_token, True, dtype=torch.bool)
            evict_mask[self.accepted_indices] = False
            batch.token_to_kv_pool_allocator.free(batch.out_cache_loc[evict_mask])
            batch.out_cache_loc = batch.out_cache_loc[self.accepted_indices]

        accept_length_list = accept_length_cpu.tolist()
        for i, req in enumerate(batch.reqs):
            req.kv_committed_len += accept_length_list[i] + 1
            req.kv_allocated_len = req.kv_committed_len

        assign_req_to_token_pool[(bs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            batch.seq_lens + self.accept_length + 1,
            batch.out_cache_loc,
            batch.req_to_token_pool.req_to_token.shape[1],
            triton.next_power_of_2(bs),
        )

    def _sample_token_from_logits(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        min_p: float,
    ) -> int:
        if temperature <= 0:
            return torch.argmax(logits).item()

        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)

        if top_k != TOP_K_ALL and top_k > 0:
            top_k = min(top_k, probs.numel())
            vals, idx = torch.topk(probs, top_k)
            mask = torch.zeros_like(probs)
            mask[idx] = vals
            probs = mask

        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cum > top_p
            cutoff[0] = False
            sorted_probs[cutoff] = 0
            probs = torch.zeros_like(probs)
            probs[sorted_idx] = sorted_probs

        if min_p > 0:
            max_prob = probs.max()
            probs = torch.where(probs >= max_prob * min_p, probs, torch.zeros_like(probs))

        if probs.sum() <= 0:
            return torch.argmax(logits).item()

        probs = probs / probs.sum()
        return torch.multinomial(probs, num_samples=1).item()

    def compute_window_preds(
        self,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
    ) -> torch.Tensor:
        sampling_info = batch.sampling_info
        logits = logits_output.next_token_logits
        needs_clone = (
            sampling_info.has_custom_logit_processor
            or sampling_info.penalizer_orchestrator.is_required
        )
        if needs_clone:
            logits = logits.clone()
            if sampling_info.has_custom_logit_processor:
                apply_custom_logit_processor(
                    logits,
                    sampling_info,
                    num_tokens_in_batch=self.draft_token_num,
                )

            if sampling_info.penalizer_orchestrator.is_required:
                linear_penalty = torch.zeros(
                    (batch.batch_size(), logits.shape[1]),
                    dtype=torch.float32,
                    device=logits.device,
                )
                sampling_info.apply_logits_bias(linear_penalty)
                logits.add_(
                    torch.repeat_interleave(
                        linear_penalty, self.draft_token_num, dim=0
                    )
                )

        bs = batch.batch_size()
        vocab_size = logits.shape[-1]
        logits = logits.reshape(bs, self.draft_token_num, vocab_size)
        window_preds = torch.argmax(logits[:, : self.lookahead_len], dim=-1)
        return window_preds

    def verify(
        self,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
        page_size: int,
    ):
        sampling_info = batch.sampling_info
        force_greedy = True
        if self.lookahead_len <= 0:
            raise ValueError("lookahead_len must be >= 1 for lookahead verify")

        if sampling_info.has_custom_logit_processor:
            apply_custom_logit_processor(
                logits_output.next_token_logits,
                sampling_info,
                num_tokens_in_batch=self.draft_token_num,
            )

        if sampling_info.penalizer_orchestrator.is_required:
            linear_penalty = torch.zeros(
                (batch.batch_size(), logits_output.next_token_logits.shape[1]),
                dtype=torch.float32,
                device=self.device,
            )
            sampling_info.apply_logits_bias(linear_penalty)
            logits_output.next_token_logits.add_(
                torch.repeat_interleave(linear_penalty, self.draft_token_num, dim=0)
            )

        bs = batch.batch_size()
        vocab_size = logits_output.next_token_logits.shape[-1]
        logits = logits_output.next_token_logits.view(bs, self.draft_token_num, vocab_size)

        draft_tokens = self.draft_token.view(bs, self.draft_token_num)
        if force_greedy or sampling_info.is_all_greedy:
            window_preds = torch.argmax(
                logits[:, : self.lookahead_len], dim=-1
            ).to(torch.int64)
        else:
            raise NotImplementedError("Non-greedy lookahead verify is not supported.")

        draft_window = draft_tokens[
            :, self.root_len : self.root_len + self.lookahead_len
        ]
        mismatch_mask = window_preds.ne(draft_window)
        has_mismatch = mismatch_mask.any(dim=-1)
        first_mismatch = mismatch_mask.float().argmax(dim=-1)

        self.accept_length = torch.where(
            has_mismatch,
            first_mismatch - 1,
            self.lookahead_len - 1,
        ).to(torch.int32)
        accept_counts = (self.accept_length + 1).clamp(min=0).to(torch.int64)

        self.mismatch_token = torch.full(
            (bs,), -1, dtype=torch.int64, device=self.device
        )
        if has_mismatch.any():
            mismatch_indices = torch.nonzero(has_mismatch, as_tuple=True)[0]
            self.mismatch_token[mismatch_indices] = window_preds[
                mismatch_indices, first_mismatch[mismatch_indices]
            ]

        self.accept_last_token = torch.empty((bs,), dtype=torch.int64, device=self.device)
        self.accept_last_token[has_mismatch] = self.mismatch_token[has_mismatch]
        self.accept_last_token[~has_mismatch] = window_preds[:, self.lookahead_len - 1]
        self.accept_last_token_cpu = self.accept_last_token.cpu()
        self.mismatch_token_cpu = self.mismatch_token.cpu()

        self.predict = self.draft_token.to(torch.int32).clone()
        if has_mismatch.any():
            base_offsets = (
                torch.arange(bs, device=self.device, dtype=torch.int64)
                * self.draft_token_num
            )
            mismatch_pos = base_offsets + self.root_len + first_mismatch
            mismatch_pos = mismatch_pos[has_mismatch]
            self.predict[mismatch_pos] = self.mismatch_token[has_mismatch].to(
                torch.int32
            )

        total_accepted = int(accept_counts.sum().item())
        if total_accepted > 0:
            base_offsets = (
                torch.arange(bs, device=self.device, dtype=torch.int64)
                * self.draft_token_num
                + self.root_len
            )
            out_offsets = torch.cumsum(accept_counts, dim=0) - accept_counts
            out_offsets_rep = out_offsets.repeat_interleave(accept_counts)
            base_offsets_rep = base_offsets.repeat_interleave(accept_counts)
            local_pos = (
                torch.arange(total_accepted, device=self.device) - out_offsets_rep
            )
            self.accepted_indices = base_offsets_rep + local_pos
            accepted_tokens = self.predict[self.accepted_indices]
        else:
            self.accepted_indices = torch.empty(
                0, device=self.device, dtype=torch.int64
            )
            accepted_tokens = torch.empty(
                0, device=self.device, dtype=torch.int32
            )

        draft_tokens_cpu = draft_tokens.cpu()
        window_preds_cpu = window_preds.cpu()
        self.predicted_pairs = []
        for i in range(bs):
            pairs: List[Tuple[int, int]] = []
            for j in range(self.lookahead_len):
                prev_token = int(draft_tokens_cpu[i, self.root_len - 1 + j].item())
                pred = int(window_preds_cpu[i, j].item())
                pairs.append((prev_token, pred))
            self.predicted_pairs.append(pairs)

        adjusted_counts = self._fill_requests(batch, accept_counts, accepted_tokens)
        adjusted_counts_gpu = adjusted_counts.to(self.device)
        self.accept_length = adjusted_counts_gpu.to(torch.int32) - 1
        accept_counts = (self.accept_length + 1).clamp(min=0).to(torch.int64)

        total_accepted = int(accept_counts.sum().item())
        if total_accepted > 0:
            base_offsets = (
                torch.arange(bs, device=self.device, dtype=torch.int64)
                * self.draft_token_num
                + self.root_len
            )
            out_offsets = torch.cumsum(accept_counts, dim=0) - accept_counts
            out_offsets_rep = out_offsets.repeat_interleave(accept_counts)
            base_offsets_rep = base_offsets.repeat_interleave(accept_counts)
            local_pos = (
                torch.arange(total_accepted, device=self.device) - out_offsets_rep
            )
            self.accepted_indices = base_offsets_rep + local_pos
        else:
            self.accepted_indices = torch.empty(
                0, device=self.device, dtype=torch.int64
            )

        if self.debug:
            def decode_text(req, tokens: List[int]) -> str:
                tokenizer = getattr(req, "tokenizer", None)
                if tokenizer is None:
                    return " ".join(str(t) for t in tokens)
                try:
                    return tokenizer.decode(tokens)
                except Exception:
                    try:
                        return " ".join(tokenizer.convert_ids_to_tokens(tokens))
                    except Exception:
                        return " ".join(str(t) for t in tokens)

            def decode_window(req, tokens: List[int]) -> str:
                tokenizer = getattr(req, "tokenizer", None)
                if tokenizer is None:
                    return " ".join(str(t) for t in tokens)
                try:
                    return " ".join(tokenizer.decode([t]) for t in tokens)
                except Exception:
                    try:
                        return " ".join(tokenizer.convert_ids_to_tokens(tokens))
                    except Exception:
                        return " ".join(str(t) for t in tokens)

            for i, req in enumerate(batch.reqs):
                prefix_tokens = req.origin_input_ids + req.output_ids
                prefix_text = decode_text(req, prefix_tokens)
                pred_tokens = window_preds_cpu[i].tolist()
                predict_window_str = decode_window(req, pred_tokens)
                draft_window = draft_tokens_cpu[
                    i, self.root_len : self.root_len + self.lookahead_len
                ].tolist()
                draft_window_str = decode_window(req, draft_window)
                logger.info(
                    'draft window tokens: "%s", predict window tokens: "%s"',
                    draft_window_str,
                    predict_window_str,
                )
                logger.info(
                    'forward result: "%s", predict window: "%s"',
                    prefix_text,
                    predict_window_str,
                )
                mismatch_pos = -1
                for j, pred in enumerate(pred_tokens):
                    if draft_window[j] != pred:
                        mismatch_pos = j
                        break
                acc_len = int(self.accept_length[i].item())
                start = int(offsets_cpu[i].item())
                end = start + int(accept_counts_cpu[i].item())
                indices = (
                    self.accepted_indices[start:end].tolist()
                    if acc_len >= 0 and end > start
                    else []
                )
                accepted_tokens = [int(self.predict[idx].item()) for idx in indices]
                accepted_strs = None
                if getattr(req, "tokenizer", None) is not None:
                    try:
                        accepted_strs = [req.tokenizer.decode([t]) for t in accepted_tokens]
                    except Exception:
                        accepted_strs = None
                # logger.info(
                #     "lookahead verify rid=%s accept_len=%d mismatch=%d "
                #     "draft_window=%s predicted=%s accepted_tokens=%s accepted_strs=%s",
                #     req.rid,
                #     acc_len,
                #     mismatch_pos,
                #     draft_window,
                #     predicted_windows[i],
                #     accepted_tokens,
                #     accepted_strs,
                # )

        if self.debug:
            for i, req in enumerate(batch.reqs):
                pred_tokens = window_preds_cpu[i].tolist()
                pred_strs = None
                if getattr(req, "tokenizer", None) is not None:
                    try:
                        pred_strs = [req.tokenizer.decode([t]) for t in pred_tokens]
                    except Exception:
                        pred_strs = None
                # Log top-k candidates for the first few draft positions
                topk_info = []
                max_pos = min(3, self.lookahead_len)
                for pos in range(max_pos):
                    logits_row = logits[i, pos]
                    vals, idxs = torch.topk(logits_row, k=5)
                    ids = idxs.tolist()
                    strs = None
                    if getattr(req, "tokenizer", None) is not None:
                        try:
                            strs = [req.tokenizer.decode([t]) for t in ids]
                        except Exception:
                            strs = None
                    topk_info.append(
                        {
                            "pos": pos,
                            "top_ids": ids,
                            "top_strs": strs,
                            "top_logits": vals.tolist(),
                        }
                    )
                # logger.info(
                #     "lookahead forward_out rid=%s predicted_tokens=%s predicted_strs=%s topk=%s",
                #     req.rid,
                #     pred_tokens,
                #     pred_strs,
                #     topk_info,
                # )

        logits_output.next_token_logits = logits_output.next_token_logits[
            self.accepted_indices
        ]
        if logits_output.hidden_states is not None:
            logits_output.hidden_states = logits_output.hidden_states[
                self.accepted_indices
            ]
        self.verified_id = self.predict[self.accepted_indices]

        accept_length_cpu = self.accept_length.cpu()
        num_accepted_tokens = (accept_length_cpu + 1).sum().item()

        self._free_cache(batch, page_size, accept_length_cpu)
        batch.seq_lens.add_(self.accept_length + 1)
        batch.seq_lens_cpu.add_(accept_length_cpu + 1)

        return logits_output, self.verified_id, num_accepted_tokens

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        pass

    def merge_batch(self, spec_info: "LookaheadVerifyInput"):
        pass

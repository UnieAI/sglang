import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.mem_cache.common import alloc_for_decode
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.lookahead_info import LookaheadVerifyInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)


@dataclass
class LookaheadState:
    window: List[int]
    req_ptr: int


class LookaheadWorker:
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
        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.model_config = target_worker.model_config
        self.tp_rank = tp_rank
        self.page_size = server_args.page_size

        self.window = server_args.lookahead_window
        self.pool_capacity = server_args.lookahead_pool_size
        self.guess_set_size = server_args.lookahead_guess_set_size
        self.pool_from_prompt = server_args.lookahead_pool_from_prompt
        self.random_prefill = server_args.lookahead_random_prefill
        self.disable_pool_update = server_args.lookahead_disable_pool_update
        self.keep_prefix_last_token = server_args.lookahead_keep_prefix_last_token
        self.prefix_only_mask = server_args.lookahead_prefix_only_mask
        self.debug = server_args.lookahead_debug
        self.jacobi_max_iter = server_args.lookahead_jacobi_max_iter
        if self.random_prefill:
            logger.warning(
                "lookahead_random_prefill is ignored; using last-token prefill."
            )

        self.root_len = 1
        self.lookahead_len = self.window
        self.bonus_len = 0
        self.draft_token_num = self.root_len + self.lookahead_len
        if server_args.speculative_num_draft_tokens != self.draft_token_num:
            logger.warning(
                "speculative_num_draft_tokens mismatch for lookahead; check server args."
            )

        self.max_batch_size = target_worker.max_running_requests
        self.device = f"cuda:{gpu_id}" if gpu_id >= 0 else self.model_runner.device
        self.vocab_size = self.model_runner.model_config.vocab_size

        self._states: Dict[str, LookaheadState] = {}
        self._pool: OrderedDict = OrderedDict()

    def _decode_tokens(self, req, tokens: List[int]) -> List[str]:
        tokenizer = getattr(req, "tokenizer", None)
        if tokenizer is None:
            return [str(t) for t in tokens]
        try:
            return [tokenizer.decode([t]) for t in tokens]
        except Exception:
            try:
                return tokenizer.convert_ids_to_tokens(tokens)
            except Exception:
                return [str(t) for t in tokens]

    def _decode_text(self, req, tokens: List[int]) -> str:
        tokenizer = getattr(req, "tokenizer", None)
        if tokenizer is None:
            return " ".join(str(t) for t in tokens)
        try:
            return tokenizer.decode(tokens)
        except Exception:
            return " ".join(self._decode_tokens(req, tokens))

    def _get_special_token_ids(self, req) -> Optional[set[int]]:
        tokenizer = getattr(req, "tokenizer", None)
        if tokenizer is None:
            return None
        ids = getattr(tokenizer, "all_special_ids", None)
        if not ids:
            return None
        try:
            return set(int(x) for x in ids)
        except Exception:
            return None

    def clear_cache_pool(self):
        self._states.clear()

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
            probs = torch.where(
                probs >= max_prob * min_p, probs, torch.zeros_like(probs)
            )

        if probs.sum() <= 0:
            return torch.argmax(logits).item()

        probs = probs / probs.sum()
        return torch.multinomial(probs, num_samples=1).item()

    def _sample_from_source(self, source_tokens: List[int], n: int) -> List[int]:
        if source_tokens:
            idx = torch.randint(
                0, len(source_tokens), (n,), device=self.device, dtype=torch.int64
            )
            src = torch.tensor(source_tokens, device=self.device, dtype=torch.int64)
            return src[idx].tolist()
        return torch.randint(
            0, self.vocab_size, (n,), device=self.device, dtype=torch.int64
        ).tolist()

    def _pool_insert(self, pair: List[int]):
        key = pair[0]
        value = pair[1]
        od = self._pool.get(key)
        if od is None:
            od = OrderedDict()
            self._pool[key] = od
        else:
            self._pool.move_to_end(key)
        if value in od:
            od.move_to_end(value)
        else:
            od[value] = None
            if len(od) > self.guess_set_size:
                od.popitem(last=False)
        if len(self._pool) > self.pool_capacity:
            self._pool.popitem(last=False)

    def _init_state(self, req, last_token: int) -> LookaheadState:
        if self.pool_from_prompt and len(req.origin_input_ids) >= 2:
            tokens = req.origin_input_ids
            for i in range(len(tokens) - 1):
                self._pool_insert([tokens[i], tokens[i + 1]])
        window = [last_token] * self.window
        return LookaheadState(window=window, req_ptr=id(req))

    def _get_state(self, req, last_token: int) -> LookaheadState:
        state = self._states.get(req.rid)
        if state is None or state.req_ptr != id(req):
            state = self._init_state(req, last_token)
            self._states[req.rid] = state
        return state

    def _sample_prefill_window(self, last_token: int) -> List[int]:
        return [last_token] * self.window

    def _build_custom_mask(self, seq_len: int) -> torch.Tensor:
        draft_len = self.draft_token_num
        total_len = seq_len + draft_len
        mask = torch.zeros(
            (draft_len, total_len), dtype=torch.bool, device=self.device
        )
        if seq_len > 0:
            mask[:, :seq_len] = True
            # Avoid double-counting the prefix last token because we re-insert it
            # as the draft root token.
            if not self.keep_prefix_last_token and not self.prefix_only_mask:
                mask[:, seq_len - 1] = False

        if not self.prefix_only_mask:
            # Strict causal chain for draft tokens appended after the prefix.
            causal = torch.tril(
                torch.ones((draft_len, draft_len), dtype=torch.bool, device=self.device)
            )
            mask[:, seq_len : seq_len + draft_len] = causal

        return mask.reshape(-1)

    def _prepare_for_speculative_decoding(self, batch: ScheduleBatch):
        if batch.forward_mode.is_extend():
            return

        bs = batch.batch_size()
        draft_tokens_list: List[List[int]] = []
        positions_list: List[List[int]] = []
        custom_mask_list: List[torch.Tensor] = []

        for i, req in enumerate(batch.reqs):
            if req.output_ids:
                last_token = req.output_ids[-1]
            elif req.origin_input_ids:
                last_token = req.origin_input_ids[-1]
            else:
                last_token = 0
            state = self._get_state(req, int(last_token))
            lookahead_tokens = list(state.window)
            if len(lookahead_tokens) != self.lookahead_len:
                lookahead_tokens = self._sample_prefill_window(int(last_token))
                state.window = list(lookahead_tokens)
            draft_tokens = [last_token] + lookahead_tokens
            draft_tokens_list.append(draft_tokens)

            if batch.seq_lens_cpu is not None:
                seq_len = int(batch.seq_lens_cpu[i])
            else:
                seq_len = len(req.origin_input_ids) + len(req.output_ids)
            positions = []
            root_pos = seq_len - 1 if seq_len > 0 else 0
            positions.append(root_pos)
            for idx in range(self.lookahead_len):
                positions.append(root_pos + 1 + idx)
            positions_list.append(positions)

            custom_mask = self._build_custom_mask(seq_len)
            custom_mask_list.append(custom_mask)

            if self.debug:
                prefix_tokens = req.origin_input_ids + req.output_ids
                prefix_text = self._decode_text(req, prefix_tokens)
                root_str = self._decode_tokens(req, [last_token])[0]
                window_strs = self._decode_tokens(req, lookahead_tokens)
                try:
                    mask_view = custom_mask.view(
                        self.draft_token_num, seq_len + self.draft_token_num
                    )
                    mask_counts = mask_view.sum(dim=1).tolist()
                except Exception:
                    mask_counts = None
                prefix_tail_ids = prefix_tokens[-8:] if prefix_tokens else []
                prefix_tail_strs = self._decode_tokens(req, prefix_tail_ids)
                root_mismatch = (
                    int(prefix_tokens[-1]) != int(last_token)
                    if prefix_tokens
                    else False
                )
                # logger.info(
                #     "lookahead forward_in rid=%s seq_len=%d kv_committed_len=%d kv_allocated_len=%d "
                #     "origin_len=%d output_len=%d root_mismatch=%s positions=%s mask_counts=%s "
                #     "prefix_text=%s root_token=%d root_str=%s prefix_tail_ids=%s prefix_tail_strs=%s "
                #     "window_tokens=%s window_strs=%s",
                #     req.rid,
                #     seq_len,
                #     int(getattr(req, "kv_committed_len", 0)),
                #     int(getattr(req, "kv_allocated_len", 0)),
                #     len(req.origin_input_ids),
                #     len(req.output_ids),
                #     root_mismatch,
                #     positions,
                #     mask_counts,
                #     prefix_text,
                #     last_token,
                #     root_str,
                #     prefix_tail_ids,
                #     prefix_tail_strs,
                #     lookahead_tokens,
                #     window_strs,
                # )
                prefill_text = self._decode_text(req, prefix_tokens + lookahead_tokens)
                logger.info(
                    'prefill from "%s" to "%s"',
                    prefix_text,
                    prefill_text,
                )
                logger.info(
                    'forward: "%s", draft window: "%s"',
                    prefix_text,
                    " ".join(window_strs),
                )

        draft_tokens_tensor = torch.tensor(
            draft_tokens_list, device=self.device, dtype=torch.int64
        ).reshape(-1)
        positions_tensor = torch.tensor(
            positions_list, device=self.device, dtype=torch.int64
        ).reshape(-1)
        custom_mask_tensor = torch.cat(custom_mask_list, dim=0)

        batch.spec_algorithm = SpeculativeAlgorithm.LOOKAHEAD
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = LookaheadVerifyInput(
            draft_token=draft_tokens_tensor,
            custom_mask=custom_mask_tensor,
            positions=positions_tensor,
            draft_token_num=self.draft_token_num,
            lookahead_len=self.lookahead_len,
            debug=self.debug,
        )
        batch.spec_info.prepare_for_verify(batch, self.page_size)

    def _update_window_and_pool(
        self,
        batch: ScheduleBatch,
        verify_input: LookaheadVerifyInput,
    ):
        if self.disable_pool_update:
            for req in batch.reqs:
                if req.finished() or req.is_retracted:
                    self._states.pop(req.rid, None)
                    continue
                if req.output_ids:
                    last_token = req.output_ids[-1]
                elif req.origin_input_ids:
                    last_token = req.origin_input_ids[-1]
                else:
                    last_token = 0
                state = self._get_state(req, int(last_token))
                state.window = [int(last_token)] * self.window
            return
        for i, req in enumerate(batch.reqs):
            if req.finished() or req.is_retracted:
                self._states.pop(req.rid, None)
                continue

            if req.output_ids:
                last_token = req.output_ids[-1]
            elif req.origin_input_ids:
                last_token = req.origin_input_ids[-1]
            else:
                last_token = 0
            state = self._get_state(req, int(last_token))
            for pair in verify_input.predicted_pairs[i]:
                self._pool_insert(list(pair))
            # Discard stale guesses after verification; rebuild window from pool.
            old_window = list(state.window)
            if len(old_window) != self.window:
                old_window = (old_window + [int(last_token)] * self.window)[: self.window]

            new_window = list(old_window)
            filled = []
            current = int(verify_input.accept_last_token[i].item())
            for pos in range(self.window):
                od = self._pool.get(current)
                if not od:
                    break
                choices = list(od.keys())
                if len(choices) == 1:
                    next_token = choices[0]
                else:
                    idx = torch.randint(
                        0, len(choices), (1,), device=self.device, dtype=torch.int64
                    ).item()
                    next_token = choices[idx]
                new_window[pos] = next_token
                filled.append((current, next_token))
                current = next_token
            state.window = new_window
            if self.debug:
                old_str = " ".join(self._decode_tokens(req, old_window))
                new_str = " ".join(self._decode_tokens(req, state.window))
                logger.info('update window: "%s" -> "%s"', old_str, new_str)
                filled_strs = []
                for src, dst in filled:
                    src_str = self._decode_tokens(req, [src])[0]
                    dst_str = self._decode_tokens(req, [dst])[0]
                    filled_strs.append(f"{src_str}->{dst_str}")
                logger.info("pool filled: %s", " | ".join(filled_strs))

    def _decode_mismatch_tokens(
        self,
        batch: ScheduleBatch,
        mismatch_tokens: torch.Tensor,
    ) -> List[int]:
        mismatch_indices = []
        mismatch_token_list = []
        for i, req in enumerate(batch.reqs):
            if req.finished() or req.is_retracted:
                continue
            token_id = int(mismatch_tokens[i].item())
            if token_id >= 0:
                mismatch_indices.append(i)
                mismatch_token_list.append(token_id)

        if not mismatch_indices:
            return []

        idxs_device = torch.tensor(
            mismatch_indices, dtype=torch.int64, device=self.device
        )

        sub_reqs = [batch.reqs[i] for i in mismatch_indices]
        sub_batch = ScheduleBatch(reqs=sub_reqs, batch_is_full=False)
        sub_batch.req_to_token_pool = batch.req_to_token_pool
        sub_batch.token_to_kv_pool_allocator = batch.token_to_kv_pool_allocator
        sub_batch.tree_cache = batch.tree_cache
        sub_batch.model_config = batch.model_config
        sub_batch.device = batch.device
        sub_batch.forward_mode = ForwardMode.DECODE
        sub_batch.enable_overlap = batch.enable_overlap
        sub_batch.has_grammar = any(req.grammar for req in sub_reqs)
        sub_batch.req_pool_indices = batch.req_pool_indices[idxs_device]
        sub_batch.seq_lens = batch.seq_lens[idxs_device]
        sub_batch.seq_lens_cpu = batch.seq_lens_cpu[mismatch_indices]
        sub_batch.orig_seq_lens = (
            batch.orig_seq_lens[idxs_device] if batch.orig_seq_lens is not None else None
        )
        sub_batch.seq_lens_sum = sub_batch.seq_lens.sum().item()
        sub_batch.return_logprob = False
        sub_batch.top_logprobs_nums = None
        sub_batch.token_ids_logprobs = None
        sub_batch.multimodal_inputs = (
            [batch.multimodal_inputs[i] for i in mismatch_indices]
            if batch.multimodal_inputs is not None
            else None
        )
        sub_batch.encoder_cached = (
            [batch.encoder_cached[i] for i in mismatch_indices]
            if batch.encoder_cached is not None
            else None
        )
        sub_batch.encoder_lens = (
            batch.encoder_lens[idxs_device]
            if batch.encoder_lens is not None
            else None
        )
        sub_batch.encoder_lens_cpu = (
            [batch.encoder_lens_cpu[i] for i in mismatch_indices]
            if batch.encoder_lens_cpu is not None
            else None
        )
        sub_batch.encoder_out_cache_loc = (
            batch.encoder_out_cache_loc[idxs_device]
            if batch.encoder_out_cache_loc is not None
            else None
        )
        sub_batch.dimensions = (
            [batch.dimensions[i] for i in mismatch_indices]
            if batch.dimensions is not None
            else None
        )
        sub_batch.hicache_consumer_index = batch.hicache_consumer_index
        sub_batch.is_prefill_only = False
        sub_batch.spec_algorithm = SpeculativeAlgorithm.NONE
        sub_batch.spec_info = None
        sub_batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            sub_batch, self.vocab_size
        )

        sub_batch.input_ids = torch.tensor(
            mismatch_token_list, device=self.device, dtype=torch.int64
        )
        sub_batch.out_cache_loc = alloc_for_decode(sub_batch, token_per_req=1)

        for req in sub_batch.reqs:
            req.decode_batch_idx += 1
            req.kv_committed_len += 1
            req.kv_allocated_len += 1

        sub_batch.seq_lens.add_(1)
        sub_batch.seq_lens_cpu.add_(1)
        if sub_batch.orig_seq_lens is not None:
            sub_batch.orig_seq_lens.add_(1)
        sub_batch.seq_lens_sum += len(sub_batch.reqs)

        self.target_worker.forward_batch_generation(
            sub_batch.get_model_worker_batch(), is_verify=True
        )

        batch.seq_lens[idxs_device] += 1
        batch.seq_lens_cpu[mismatch_indices] += 1
        if batch.orig_seq_lens is not None:
            batch.orig_seq_lens[idxs_device] += 1
        batch.seq_lens_sum = batch.seq_lens.sum().item()

        for idx, token_id in zip(mismatch_indices, mismatch_token_list):
            req = batch.reqs[idx]
            req.output_ids.append(token_id)
            if req.grammar is not None:
                try:
                    req.grammar.accept_token(token_id)
                except ValueError as e:
                    logger.info(f"{idx=}, {req=}")
                    raise e
            req.check_finished()

        return mismatch_indices

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        if batch.forward_mode.is_extend():
            model_worker_batch = batch.get_model_worker_batch()
            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch
            )
            logits_output, next_token_ids, can_run_cuda_graph = (
                batch_result.logits_output,
                batch_result.next_token_ids,
                batch_result.can_run_cuda_graph,
            )
            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_accepted_tokens=0,
                can_run_cuda_graph=can_run_cuda_graph,
            )

        self._prepare_for_speculative_decoding(batch)
        verify_input: LookaheadVerifyInput = batch.spec_info
        logits_output = None
        can_run_cuda_graph = False

        jacobi_iters = max(1, int(self.jacobi_max_iter))
        bs = batch.batch_size()
        for it in range(jacobi_iters):
            model_worker_batch = batch.get_model_worker_batch()
            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch, is_verify=True
            )
            logits_output = batch_result.logits_output
            can_run_cuda_graph = batch_result.can_run_cuda_graph

            if jacobi_iters == 1:
                break

            window_preds = verify_input.compute_window_preds(batch, logits_output)
            preds_tensor = torch.tensor(
                window_preds, device=self.device, dtype=torch.int64
            )
            draft_view = verify_input.draft_token.view(bs, self.draft_token_num)
            draft_window = draft_view[
                :, self.root_len : self.root_len + self.lookahead_len
            ]
            if torch.equal(draft_window, preds_tensor):
                if self.debug:
                    logger.info("lookahead jacobi converged at iter=%d", it + 1)
                break
            if it + 1 >= jacobi_iters:
                break
            draft_window.copy_(preds_tensor)

        logits_output, next_token_ids, num_accepted_tokens = verify_input.verify(
            batch, logits_output, self.page_size
        )
        self._update_window_and_pool(batch, verify_input)
        mismatch_indices = self._decode_mismatch_tokens(
            batch, verify_input.mismatch_token
        )

        # Build per-request next_token_ids (accepted draft tokens + mismatch token if any)
        accept_counts = (verify_input.accept_length + 1).clamp(min=0).tolist()
        accepted_flat = (
            next_token_ids.tolist() if next_token_ids is not None else []
        )
        offset = 0
        final_tokens: List[int] = []
        mismatch_set = set(mismatch_indices)
        for i, count in enumerate(accept_counts):
            if count > 0:
                final_tokens.extend(accepted_flat[offset : offset + count])
                offset += count
            if i in mismatch_set:
                final_tokens.append(int(verify_input.mismatch_token[i].item()))

        next_token_ids = torch.tensor(
            final_tokens, device=self.device, dtype=torch.int64
        )

        batch.forward_mode = ForwardMode.DECODE

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            num_accepted_tokens=num_accepted_tokens,
            can_run_cuda_graph=can_run_cuda_graph,
        )

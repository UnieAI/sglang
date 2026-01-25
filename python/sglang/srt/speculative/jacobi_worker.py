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

        self._init_dsc()
        logger.info("JacobiWorker initialized with DSC support")

    def _init_dsc(self):
        # DSC parameters
        self.max_batch_size = self.server_args.jacobi_max_batch_size
        self.accept_rate_low = self.server_args.jacobi_accept_rate_low
        self.accept_rate_high = self.server_args.jacobi_accept_rate_high
        self.ema_decay = self.server_args.jacobi_accept_rate_ema_decay
        self.warmup_steps = self.server_args.jacobi_accept_rate_warmup
        self.probe_interval = self.server_args.jacobi_accept_rate_probe_interval

        # DSC state
        self.current_ema = 1.0  # Optimistic initial value
        self.step_counter = 0
        self.gate_open = True
        self.probe_counter = 0

    def _use_jacobi_for_decode(self, batch) -> bool:
        # 1. Batch size gate
        if self.max_batch_size is not None and len(batch.reqs) > self.max_batch_size:
            return False

        # 2. Acceptance rate gate
        # If low/high thresholds are not set, always open
        if self.accept_rate_low is None:
            return True

        self.step_counter += 1

        # Warmup period
        if self.step_counter < self.warmup_steps:
            return True

        # Hysteresis logic
        if self.gate_open:
            if self.current_ema < self.accept_rate_low:
                logger.info(f"[DSC] Closing Jacobi gate. EMA: {self.current_ema:.3f} < {self.accept_rate_low}")
                self.gate_open = False
        else:
            if self.accept_rate_high is not None and self.current_ema > self.accept_rate_high:
                logger.info(f"[DSC] Re-opening Jacobi gate. EMA: {self.current_ema:.3f} > {self.accept_rate_high}")
                self.gate_open = True

            # Probe logic
            if not self.gate_open and self.probe_interval > 0:
                self.probe_counter += 1
                if self.probe_counter >= self.probe_interval:
                    self.probe_counter = 0
                    return True # One-shot probe

        return self.gate_open

    def _is_greedy_sampling(self, batch):
        for req in batch.reqs:
            if req.sampling_params.top_p < 1.0:
                return False
            if req.sampling_params.top_k != 1:
                return False
            if req.sampling_params.min_p > 0.0:
                return False
            if req.sampling_params.frequency_penalty != 0.0:
                return False
            if req.sampling_params.presence_penalty != 0.0:
                return False
            if req.sampling_params.repetition_penalty != 1.0:
                return False
            if req.sampling_params.logit_bias:
                return False
            if req.sampling_params.n != 1:
                return False
        return True

    def _update_acceptance_rate(self, num_accepted: int, total_proposed: int):
        if total_proposed == 0:
            return

        current_rate = num_accepted / total_proposed
        self.current_ema = self.ema_decay * self.current_ema + (1.0 - self.ema_decay) * current_rate

    def _get_stable_uncached_lens(self, batch):
        stable_uncached_lens = []
        for req in batch.reqs:
            prefix_len = len(req.prefix_indices)
            stable_len = len(req.origin_input_ids) + len(req.output_ids)
            stable_uncached_lens.append(max(stable_len - prefix_len, 0))
        return stable_uncached_lens

    def forward_batch_generation(
        self,
        model_worker_batch,
        forward_batch: Optional[ForwardBatch] = None,
        **_,
    ) -> GenerationBatchResult:
        if forward_batch is None:
            forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)

        # === DSC Check ===
        use_jacobi = self._use_jacobi_for_decode(model_worker_batch)
        if use_jacobi and not self._is_greedy_sampling(model_worker_batch):
            logger.debug("Jacobi disabled: non-greedy sampling detected")
            use_jacobi = False
        stable_uncached_lens = self._get_stable_uncached_lens(model_worker_batch)
        if use_jacobi and any(length != 1 for length in stable_uncached_lens):
            logger.debug("Jacobi disabled: stable_uncached_len != 1 detected")
            use_jacobi = False

        if not use_jacobi:
            # Fallback to standard decode: Just 1 forward step with no draft/verify loop
            logits_output, can_run_cuda_graph = self.model_runner.forward(
                forward_batch, pp_proxy_tensors=None
            )
            logits = logits_output.full_logits
            if logits is None:
                raise RuntimeError("Jacobi fallback requires full_logits but got None.")
            if logits.dim() == 3:
                logits = logits[0]
            next_token_ids = []
            logit_offset = 0
            for i in range(len(model_worker_batch.reqs)):
                if model_worker_batch.extend_seq_lens:
                    extend_len = model_worker_batch.extend_seq_lens[i]
                else:
                    extend_len = forward_batch.seq_lens[i].item()
                req_logits = logits[logit_offset : logit_offset + extend_len]
                logit_offset += extend_len
                stable_uncached_len = stable_uncached_lens[i] if i < len(stable_uncached_lens) else 1
                if stable_uncached_len > 0:
                    idx = min(stable_uncached_len - 1, req_logits.shape[0] - 1)
                else:
                    idx = req_logits.shape[0] - 1
                next_token_ids.append(int(torch.argmax(req_logits[idx], dim=-1).item()))
            next_token_ids = torch.tensor(
                next_token_ids, dtype=torch.long, device=self.device
            )
            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                can_run_cuda_graph=can_run_cuda_graph,
                force_disable_spec=True # Marker for metrics
            )
        # =================

        req_states = []

        # Initialize per-request state
        input_offset = 0
        original_inputs = forward_batch.input_ids

        for i, req in enumerate(model_worker_batch.reqs):
            if model_worker_batch.extend_seq_lens:
                extend_len = model_worker_batch.extend_seq_lens[i]
            else:
                extend_len = forward_batch.seq_lens[i].item()

            prefix_len = len(req.prefix_indices)
            stable_len = len(req.origin_input_ids) + len(req.output_ids)
            stable_uncached_len = max(stable_len - prefix_len, 0)
            draft_len = extend_len - stable_uncached_len
            if draft_len <= 0:
                draft_len = len(req.jacobi_draft_ids) or (self.block_size * self.num_blocks)
            logit_start = max(stable_uncached_len - 1, 0)

            # Extract stable prefix for this request
            stable_prefix = None
            if stable_uncached_len > 0:
                # Ensure we slice from the *original* input properly
                # We assume forward_batch.input_ids has correct layout [req1, req2...]
                stable_prefix = original_inputs[input_offset : input_offset + stable_uncached_len].clone()

            input_offset += extend_len

            # Initialize draft
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

            req_states.append({
                "req": req,
                "draft_len": draft_len,
                "stable_uncached_len": stable_uncached_len,
                "stable_prefix": stable_prefix,
                "logit_start": logit_start,
                "new_draft": draft_tensor,
                "accepted_ids": torch.empty((0,), dtype=torch.long, device=self.device),
                "bootstrap": req.jacobi_needs_bootstrap
            })

        can_run_cuda_graph = False
        logits_output = None
        steps = max(self.steps_per_yield, 1)

        # Metrics for DSC
        total_accepted_tokens = 0
        total_drafted_tokens = 0

        for _ in range(steps):
            # Construct batched inputs
            batch_inputs = []
            for state in req_states:
                if state["stable_prefix"] is None:
                    batch_inputs.append(state["new_draft"])
                else:
                    batch_inputs.append(torch.cat([state["stable_prefix"], state["new_draft"]], dim=0))

            batched_input_tensor = torch.cat(batch_inputs, dim=0)

            if forward_batch.input_ids.numel() == batched_input_tensor.numel():
                forward_batch.input_ids.copy_(batched_input_tensor)
            else:
                forward_batch.input_ids = batched_input_tensor

            logits_output, can_run_cuda_graph = self.model_runner.forward(
                forward_batch, pp_proxy_tensors=None
            )

            logits = logits_output.full_logits
            if logits is None:
                raise RuntimeError("Jacobi requires full_logits but got None.")
            if logits.dim() == 3:
                logits = logits[0]

            # Process each request's logits
            logit_offset = 0
            for state in req_states:
                req = state["req"]
                draft_len = state["draft_len"]
                total_len = state["stable_uncached_len"] + draft_len

                # Slice logits for this request
                req_logits = logits[logit_offset : logit_offset + total_len]
                logit_offset += total_len

                logit_start = state["logit_start"]
                verification_logits = req_logits[logit_start : logit_start + draft_len]

                if state["bootstrap"]:
                    state["new_draft"] = torch.argmax(verification_logits, dim=-1)
                    state["accepted_ids"] = torch.empty((0,), dtype=torch.long, device=self.device)
                    state["bootstrap"] = False
                    continue

                # Verification Logic
                current_draft = state["new_draft"]
                greedy_tokens = torch.argmax(verification_logits, dim=-1)

                # Truncate to min length comparing (current_draft vs greedy_tokens)
                # current_draft len N. greedy_tokens len N (if S_last was included and draft_len is N).
                check_len = min(current_draft.numel(), greedy_tokens.numel())

                mismatch = current_draft[:check_len] != greedy_tokens[:check_len]
                # Find first mismatch index
                if mismatch.any():
                    accepted_len = int(torch.argmax(mismatch.int()).item())
                else:
                    accepted_len = check_len

                state["accepted_ids"] = current_draft[:accepted_len]

                had_rejection = accepted_len < current_draft.numel()
                total_accepted_tokens += accepted_len

                # Next draft generation
                next_token_logit = req_logits[logit_start + accepted_len]
                next_token = torch.argmax(next_token_logit, dim=-1)

                if accepted_len < current_draft.numel():
                    tail_predictions_logits = req_logits[
                        logit_start + accepted_len + 1 : logit_start + draft_len
                    ]
                    if tail_predictions_logits.numel() > 0:
                        greedy_tail = torch.argmax(tail_predictions_logits, dim=-1)
                        state["new_draft"] = torch.cat([next_token.view(1), greedy_tail], dim=0)
                    else:
                        state["new_draft"] = next_token.view(1)
                else:
                    state["new_draft"] = next_token.view(1)

                pool_tail = self._select_ngram_tail(req, draft_len - 1)
                if had_rejection and pool_tail is not None:
                    state["new_draft"] = torch.tensor(
                        [int(next_token.item())] + pool_tail,
                        dtype=torch.long,
                        device=self.device,
                    )

                if state["new_draft"].numel() < draft_len:
                    pad_len = draft_len - state["new_draft"].numel()
                    pad_token = state["new_draft"][-1]
                    pad_tokens = pad_token.repeat(pad_len)
                    state["new_draft"] = torch.cat([state["new_draft"], pad_tokens], dim=0)
                elif state["new_draft"].numel() > draft_len:
                    state["new_draft"] = state["new_draft"][:draft_len]

                if self.ngram_pool_size > 0:
                    self._append_ngram_pool(req, state["new_draft"].tolist())

                total_drafted_tokens += draft_len # Approximation, per step

        # Update DSC Stats
        if self.accept_rate_low is not None:
            self._update_acceptance_rate(total_accepted_tokens, total_drafted_tokens)

        # Aggregate results
        flat_next_token_ids = []
        accept_lengths = []
        all_draft_ids = []

        for state in req_states:
            accepted_ids = state["accepted_ids"]
            accept_lengths.append(accepted_ids.numel())
            if accepted_ids.numel() > 0:
                flat_next_token_ids.append(accepted_ids)
            all_draft_ids.append(state["new_draft"].tolist())

        if flat_next_token_ids:
            next_token_ids = torch.cat(flat_next_token_ids, dim=0)
        else:
            next_token_ids = torch.empty((0,), dtype=torch.long, device=self.device)

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            draft_ids=all_draft_ids,
            num_accepted_tokens=sum(accept_lengths),
            accept_length_per_req_cpu=accept_lengths,
            can_run_cuda_graph=can_run_cuda_graph,
        )

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

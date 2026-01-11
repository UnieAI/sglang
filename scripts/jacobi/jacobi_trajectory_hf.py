from __future__ import annotations

from typing import Optional, Tuple

import torch


def _infer_seq_dim(
    key: torch.Tensor,
    num_heads: Optional[int],
    head_dim: Optional[int],
) -> int:
    if key.dim() < 3:
        return -1
    if head_dim is not None and key.shape[-1] == head_dim:
        if num_heads is not None and key.dim() >= 4 and key.shape[-2] == num_heads:
            return -3
        return -2
    if num_heads is not None and key.dim() >= 4 and key.shape[1] == num_heads:
        return 2
    return -2


def _get_head_dims(model) -> Tuple[Optional[int], Optional[int]]:
    num_heads = getattr(model.config, "num_attention_heads", None)
    head_dim = getattr(model.config, "head_dim", None)
    if head_dim is None and num_heads:
        head_dim = model.config.hidden_size // num_heads
    return num_heads, head_dim


def _get_cache_len(
    past_key_values,
    num_heads: Optional[int],
    head_dim: Optional[int],
) -> int:
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return past_key_values.get_seq_length()
    if isinstance(past_key_values, (list, tuple)) and past_key_values:
        key = past_key_values[0][0]
        if not isinstance(key, torch.Tensor):
            return 0
        seq_dim = _infer_seq_dim(key, num_heads, head_dim)
        if seq_dim == -1:
            return 0
        return key.shape[seq_dim]
    return 0


def _crop_past_key_values(
    past_key_values,
    max_length: int,
    num_heads: Optional[int],
    head_dim: Optional[int],
):
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "crop"):
        past_key_values.crop(max_length)
        return past_key_values
    if not isinstance(past_key_values, (list, tuple)) or not past_key_values:
        return past_key_values

    key = past_key_values[0][0]
    if not isinstance(key, torch.Tensor):
        return past_key_values

    seq_dim = _infer_seq_dim(key, num_heads, head_dim)
    if seq_dim == -1:
        return past_key_values
    seq_dim = seq_dim % key.dim()

    new_past = []
    for layer in past_key_values:
        if len(layer) < 2:
            new_past.append(layer)
            continue
        k, v, *rest = layer
        if isinstance(k, torch.Tensor) and max_length < k.shape[seq_dim]:
            k = k.narrow(seq_dim, 0, max_length)
        if isinstance(v, torch.Tensor) and max_length < v.shape[seq_dim]:
            v = v.narrow(seq_dim, 0, max_length)
        new_past.append((k, v, *rest))
    return tuple(new_past)


@torch.inference_mode()
def get_jacobi_forward_trajectory_greedy_hf(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values=None,
    use_cache: Optional[bool] = None,
    prefill_phase: Optional[bool] = False,
    n_token_seq_len: int = 64,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
    lenience: float = 1.0,
    accept_threshold: float = 0.99,
    tokenizer=None,
    eos_token_id: Optional[int] = None,
):
    if input_ids is None:
        raise ValueError("You must specify exactly input_ids")

    if prefill_phase:
        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = outputs.logits
        first_correct_token = torch.argmax(
            logits[:, -1, :], dim=-1, keepdim=True
        )
        return outputs.past_key_values, first_correct_token

    if past_key_values is None:
        raise ValueError("past_key_values must be provided for Jacobi generation.")

    num_heads, head_dim = _get_head_dims(self)
    batch, out, device = input_ids.shape[0], input_ids, input_ids.device
    accepted_n_gram = out
    answer_trajectory_ids = [out]
    total_accepted = 0

    while total_accepted < n_token_seq_len:
        base_len = _get_cache_len(past_key_values, num_heads, head_dim)
        outputs = self(
            input_ids=out,
            attention_mask=torch.ones_like(out, device=device),
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = outputs.logits
        past_key_values = outputs.past_key_values

        p_prob = torch.nn.functional.softmax(logits, dim=-1)
        greedy_tokens = torch.argmax(p_prob[:, :-1, :], dim=-1)
        mismatch = out[:, 1:] != greedy_tokens
        accepted = (mismatch.cumsum(dim=-1) == 0).sum(dim=-1) + 1
        num_accepted = int(accepted[0])
        accepted_n_gram[:, total_accepted : total_accepted + num_accepted] = (
            out[:, :num_accepted].clone()
        )
        total_accepted += num_accepted

        out_len = out.shape[1]
        has_rejected = num_accepted < out_len
        if has_rejected:
            desired_len = base_len + num_accepted
            past_key_values = _crop_past_key_values(
                past_key_values, desired_len, num_heads, head_dim
            )
            next_token = torch.argmax(
                p_prob[:, num_accepted - 1, :], dim=-1, keepdim=True
            )
            out = next_token
            q_probs_rem = p_prob[:, num_accepted:-1, :]
            if q_probs_rem.shape[1] > 0:
                q_sampled = torch.argmax(q_probs_rem, dim=-1)
                out = torch.cat((out, q_sampled), dim=-1)
                answer_trajectory_ids.append(
                    torch.cat((accepted_n_gram[:, :total_accepted], out), dim=-1)
                )
            continue

        next_token = torch.argmax(p_prob[:, -1, :], dim=-1, keepdim=True)
        total_accepted += 1
        answer_trajectory_ids.append(accepted_n_gram)

    return past_key_values, next_token, answer_trajectory_ids

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import torch
import random
import argparse
from types import MethodType
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import torch.nn.functional as F
from einops import rearrange
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

import re
from transformers.cache_utils import DynamicCache

from pathlib import Path
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from jacobi_trajectory_hf import get_jacobi_forward_trajectory_greedy_hf
from qwen2_modeling_jacobi_forcing_greedy import get_jacobi_forward_trajectory_greedy

Qwen2ForCausalLM.get_jacobi_forward_trajectory_greedy = get_jacobi_forward_trajectory_greedy


def build_prompt(tokenizer, user_text, system_prompt, use_chat_template):
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_text})
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    if system_prompt:
        return f"{system_prompt}\n{user_text}"
    return user_text

# UTILS
def load_prompt_list(filename, start=0, end=None):
    with open(filename, "r", encoding="utf-8") as f:
        data_dict = json.load(f)
    # if not isinstance(data_dict, dict):
    #     raise ValueError(f"Expected JSON object in {filename}")

    # Optionally filter the data_dict here

    end = len(data_dict) if end is None else min(end, len(data_dict))
    selected_data = data_dict[start:end]
    prompt_list = []
    for data in selected_data:
        prompt_list.append(data)
    return prompt_list

def trim_left_padding(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    assert input_ids.dim() == 2 and input_ids.size(0) == 1
    input_ids_flat = input_ids[0]
    first_non_pad = (input_ids_flat != pad_token_id).nonzero(as_tuple=True)[0][0].item()
    return input_ids[:, first_non_pad:]

def make_left_pad_attention_mask(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    is_pad = input_ids == pad_token_id
    first_non_pad_idx = (~is_pad).float().argmax(dim=1)
    seq_len = input_ids.size(1)
    position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
    return (position_ids >= first_non_pad_idx.unsqueeze(1)).long()

def compute_left_pad_lengths(batch_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    return (batch_ids != pad_token_id).float().argmax(dim=1)

def find_first_true_index(bool_tensor, dim=-1):
    return (bool_tensor.cumsum(dim=dim) == 0).sum(dim=dim)

# MAIN LOOP
def main(
    filename,
    model,
    tokenizer,
    n_token_seq_len,
    max_new_seq_len,
    use_labels,
    data_bos_id,
    data_eos_id,
    batch_size,
    save_path,
    system_prompt,
    use_chat_template,
):

    # Parse bucket_{bucket_id} from filename
    m = re.search(r"bucket_(\d+)", filename)
    if m:
        bucket_id = m.group(1)
    else:
        print(f"Warning: Could not parse bucket ID from filename '{filename}'. Using 'unknown'.")
        bucket_id = "unknown"
    
    # fixed to 0~25000 to initially load all data
    data = load_prompt_list(filename, start=0, end=25000)
    data_eos_id = min(len(data), int(data_eos_id))
    new_data = []

    for start_idx in tqdm(range(int(data_bos_id), int(data_eos_id), batch_size)):
        end_idx = min(start_idx + batch_size, int(data_eos_id))
        batch_indices = torch.arange(start_idx, end_idx, device=model.device)

        print(f"\nProcessing batch from {start_idx} to {end_idx}...\n")

        prompts = [
            build_prompt(
                tokenizer,
                data[i - int(data_bos_id)],
                system_prompt,
                use_chat_template,
            )
            for i in batch_indices
        ]

        model_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        input_ids = model_inputs["input_ids"]
        generated_ids = input_ids
        attention_mask = model_inputs["attention_mask"] 
        iterations = torch.zeros(len(batch_indices), dtype=torch.int, device=model.device)

        prefill_phase = True

        dict_lst = []
        while True:
            generated_part = generated_ids[:, model_inputs["input_ids"].size(1):]
            eos_found = (generated_part == tokenizer.eos_token_id).any(dim=1)
            still_active = ~eos_found
            if still_active.sum() == 0:
                break
            if (iterations[still_active][0] * n_token_seq_len) > max_new_seq_len:
                break

            input_ids_active = input_ids[still_active]
            attn_mask_active = make_left_pad_attention_mask(input_ids_active, tokenizer.pad_token_id)

            batch_indices_active = batch_indices[still_active]
            iterations_active = iterations[still_active]

            print(f'performing diffusion decoding for iterations: {iterations_active}', flush=True)
            if prefill_phase:
                past_key_values, first_correct_token = model.get_jacobi_forward_trajectory_greedy(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=None,
                    use_cache=True,
                    prefill_phase=prefill_phase,
                    n_token_seq_len=n_token_seq_len,
                    tokenizer=tokenizer,
                    )
                print(f'finishing prefilling...', flush=True)
                prefill_phase = False
                continue
            else:
                q_sampled = random.choices(generated_ids[0].tolist(), k=n_token_seq_len-1)
                q_sampled = torch.tensor(q_sampled, dtype=torch.long, device=model.device).unsqueeze(0)
                input_ids = torch.cat((first_correct_token.view(1,-1), q_sampled),dim=-1)
                past_key_values, first_correct_token, answer_trajectory_ids_active = model.get_jacobi_forward_trajectory_greedy(
                    input_ids=input_ids,
                    attention_mask=None,
                    past_key_values=past_key_values,
                    use_cache=True,
                    prefill_phase=prefill_phase,
                    n_token_seq_len=n_token_seq_len,
                    tokenizer=tokenizer,
                    )
                # print(f'len(answer_trajectory_ids_active): {len(answer_trajectory_ids_active)}')
                # for i in range(len(answer_trajectory_ids_active)):
                #     print(answer_trajectory_ids_active[i].shape)
                generated_ids = torch.cat((generated_ids, answer_trajectory_ids_active[-1]), dim=-1)

            for n, idx in enumerate(batch_indices_active):
                traj = answer_trajectory_ids_active
                teacher_output_ids = generated_ids
                ## Check quality ###
                generated_str = ''.join(tokenizer.decode(teacher_output_ids[0, :], skip_special_tokens=False))
                print(f'Generated answers: {generated_str}')
                ### Check quality ###
                dic = {
                    "diffusion_itr_id": f"itr_{iterations_active[n].item()}",
                    "data_id": f"bucket_{bucket_id}_data_{idx.item()}",
                    "prompt_ids": generated_ids[:, :-n_token_seq_len].cpu(),
                    "answer_trajectory_ids": [step[0].cpu() for step in traj],
                    "teacher_output_ids": teacher_output_ids[0].cpu()
                }
                iterations_active[n] += 1
                dict_lst.append(dic)
                
            batch_indices = batch_indices_active
            iterations = iterations_active

        print(f'finishing diffusion decoding...', flush=True)
        grouped_by_data_id = defaultdict(list)
        for dic in dict_lst:
            grouped_by_data_id[dic["data_id"]].append(dic)

        for data_id, group in grouped_by_data_id.items():
            best_teacher_output = max(group, key=lambda x: len(x["teacher_output_ids"]))["teacher_output_ids"]
            for dic in group:
                dic["teacher_output_ids"] = best_teacher_output
                # Now convert to list for JSON
                dic["prompt_ids"] = dic["prompt_ids"].tolist()
                dic["answer_trajectory_ids"] = [a.tolist() for a in dic["answer_trajectory_ids"]]
                dic["teacher_output_ids"] = dic["teacher_output_ids"].tolist()
                new_data.append(dic)

        os.makedirs(save_path, exist_ok=True)
        new_file_name = f"{Path(filename).stem}_greedy_jacobi_len{n_token_seq_len}_labels_{use_labels}_maxlen{max_new_seq_len}_{data_bos_id}_{data_eos_id}.json"
        new_file_path = os.path.join(save_path, new_file_name)
    
        with open(new_file_path, "w") as f:
            json.dump(new_data, f)

# ---------------- ENTRY -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--n_token_seq_len", type=int, default=64)
    parser.add_argument("--max_new_seq_len", type=int, default=16384)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data_start_id", default=0)
    parser.add_argument("--data_bos_id", default=0)
    parser.add_argument("--data_eos_id", default=40)
    parser.add_argument("--use_labels", action="store_true")
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="",
        help="Optional system prompt for chat templates.",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Disable tokenizer chat template usage.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trust_remote_code when loading models.",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="flash_attention_2",
        help="Attention backend for HF models (e.g., flash_attention_2, sdpa, eager).",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype for model loading.",
    )
    parser.add_argument(
        "--cache-implementation",
        type=str,
        default="dynamic",
        help="HF cache implementation (e.g., dynamic, static).",
    )
    args = parser.parse_args()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.torch_dtype]

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="cuda",
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token

    if hasattr(model.config, "cache_implementation"):
        model.config.cache_implementation = args.cache_implementation

    if not hasattr(model, "get_jacobi_forward_trajectory_greedy"):
        model.get_jacobi_forward_trajectory_greedy = MethodType(
            get_jacobi_forward_trajectory_greedy_hf, model
        )

    main(
        args.filename,
        model,
        tokenizer,
        args.n_token_seq_len,
        args.max_new_seq_len,
        args.use_labels,
        args.data_bos_id,
        args.data_eos_id,
        args.batch_size,
        args.save_path,
        args.system_prompt,
        not args.no_chat_template,
    )

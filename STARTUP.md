# Jacobi 启动手册（SGLang 内置训练＋推理）

本手册只使用 SGLang 仓库内的脚本与参数，不依赖 `JacobiForcing/` 目录。

---

## 0) 前置条件
- GPU 环境与 CUDA 可用。
- 安装训练依赖（若只推理可跳过）：  
  ```bash
  pip install -r scripts/jacobi/requirements.txt
  ```
  说明：训练脚本依赖 `transformers / datasets / peft / accelerate / wandb`。
- Jacobi 仅支持以下 causal LM 架构：
  `Qwen2 / Qwen3 / Llama / Mistral / Mixtral / Gemma / GptOss`

---

## 1) 训练流程（全部在 SGLang 仓库内完成）
训练流程包含 3 个阶段，全部脚本位于 `scripts/jacobi/`。

### 1.1 轨迹生成（Choice B Step 1）
```bash
python3 scripts/jacobi/generate_trajectory_opencodeinstruct_greedy.py \
  --filename /path/to/bucket_0030.json \
  --model /path/or/hf-id \
  --n_token_seq_len 32 \
  --max_new_seq_len 1024 \
  --data_bos_id 0 \
  --data_eos_id 5000 \
  --batch_size 1 \
  --save_path /path/to/output_dir \
  --system-prompt "You are a helpful assistant." \
  --attn-implementation flash_attention_2 \
  --torch-dtype bfloat16
```

补充参数：
- `--trust-remote-code`：GPT‑OSS / 私有模型需要时开启。
- `--no-chat-template`：禁用 tokenizer chat template。

输出：在 `--save_path` 生成轨迹 JSON 文件。

### 1.2 数据打包（Choice B Step 2）
```bash
python3 scripts/jacobi/prepare_training_data_progressive_noise_window.py \
  --input_path /path/to/trajectory.json \
  --output_path /path/to/output_training_seq.jsonl \
  --n_token_seq_length 32 \
  --window_size 16 \
  --min_noisy_ratio 0 \
  --max_noisy_ratio 1.0 \
  --strategy progressive
```

### 1.3 训练（Jacobi Forcing）
```bash
python3 scripts/jacobi/soft_flexattn_train_cllm_multiblock.py \
  --target_model_path /path/or/hf-id \
  --data_path /path/to/output_training_seq.jsonl \
  --output_dir /path/to/checkpoints \
  --max_new_tokens 32 \
  --bf16 True \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 1
```

说明：
- `--max_new_tokens` 必须与 `--n_token_seq_len` 相同。
- 需要多卡训练时可用 `torchrun` 包裹训练脚本。

---

## 2) SGLang 推理启动（Jacobi 解码）

### 2.1 最小启动
```bash
python -m sglang.launch_server \
  --model-path <hf_model_or_local_path> \
  --speculative-algorithm JACOBI \
  --speculative-num-draft-tokens 64 \
  --port 30000
```

### 2.2 启用三项关键特性
```bash
python -m sglang.launch_server \
  --model-path <hf_model_or_local_path> \
  --speculative-algorithm JACOBI \
  --speculative-num-draft-tokens 64 \
  --jacobi-num-blocks 2 \
  --jacobi-ngram-pool-size 4 \
  --jacobi-prefill-random \
  --jacobi-steps-per-yield 1 \
  --port 30000
```

参数说明：
- `--speculative-num-draft-tokens`：单 block 长度（n）。
- `--jacobi-num-blocks`：并行 blocks 数（K）。
- `--jacobi-ngram-pool-size`：Request 级别 n‑gram pool 大小（0 关闭）。
- `--jacobi-prefill-random`：prefill 阶段使用随机 token 初始化 draft。

---

## 3) 客户端验证
```python
import sglang as sgl

sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))
print(sgl.gen("def fibonacci(n):", max_tokens=64))
```

---

## 4) 限制与注意事项
- Jacobi 不支持 return_logprob 与 grammar 约束；streaming 已支持（按 `stream_interval` 进行分段输出）。
- MoE 模型建议先用 `--jacobi-num-blocks 1` 做稳定性验证。
- OOM 优先降低 `--speculative-num-draft-tokens` 或 `--jacobi-num-blocks`。

# UnieInfra

[![Website](https://img.shields.io/badge/website-unieai.com-0b7285?style=flat&logo=google-chrome&logoColor=white)](https://unieai.com)
[![Discord](https://img.shields.io/badge/discord-join-5865F2?style=flat&logo=discord&logoColor=white)](https://discord.gg/D4eFuB86bF)
[![GitHub Stars](https://img.shields.io/github/stars/unieai/sglang?style=flat)](https://github.com/unieai/sglang/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/unieai/sglang?style=flat)](https://github.com/unieai/sglang/forks)
[![GitHub Issues](https://img.shields.io/github/issues/unieai/sglang?style=flat)](https://github.com/unieai/sglang/issues)
[![GitHub PRs](https://img.shields.io/github/issues-pr/unieai/sglang?style=flat)](https://github.com/unieai/sglang/pulls)
[![Release](https://img.shields.io/github/v/release/unieai/sglang?style=flat)](https://github.com/unieai/sglang/releases)
[![License: NonCommercial](https://img.shields.io/badge/License-NonCommercial-lightgrey?style=flat)](https://github.com/unieai/sglang/blob/main/LICENSE)
[![CI](https://github.com/unieai/sglang/actions/workflows/pr-test.yml/badge.svg)](https://github.com/unieai/sglang/actions/workflows/pr-test.yml)

[English](README.md) | 中文

## 目錄

- [概述](#概述)
- [快速開始](#快速開始)
- [預設配置](#預設配置)
- [核心概念](#核心概念)
- [架構](#架構)
- [配置指南](#配置指南)
- [使用範例](#使用範例)
- [整合](#整合)
- [理解 Gate 機制](#理解-gate-機制)
- [常見問題](#常見問題)
- [效能建議](#效能建議)
- [故障排除](#故障排除)
- [實作細節](#實作細節)
- [開發路線圖](#開發路線圖)
- [優缺點權衡](#優缺點權衡)

## 概述

### UnieInfra 是什麼？

UnieInfra 是 SGLang 中 NGRAM 推測解碼（speculative decoding）的**動態策略控制器（Dynamic Strategy Controller, DSC）**。它會在每個 decode step 智慧地決定要使用 NGRAM 推測解碼，還是回退到標準 decode 路徑，決策依據為即時運行條件。

### 核心特性

- **逐步決策機制**：在每個 decode step 重新評估條件，無需重啟 worker
- **多重 gate 機制**：batch size、序列長度、輸出長度、接受率 gating
- **保守安全**：當 NGRAM 可能效率不佳時自動回退到標準 decode
- **零開銷回退**：使用現有的 decode 邏輯和 KV cache 管理
- **實驗友好**：所有行為透過 CLI 參數控制
- **PD 兼容**：與 Prefill/Decode 分離架構無縫配合

### 為什麼要使用 UnieInfra？

NGRAM 推測解碼在低 batch size 和高接受率時可提供顯著加速。然而，在以下情況下可能比標準 decode 更慢：
- Batch size 較大（高併發）
- 接受率較低（prompt/任務匹配度差）
- 序列非常長
- 輸出要求非常長

**沒有 UnieInfra** 時，NGRAM 即使在效率不佳時也會持續運行，可能降低整體吞吐量。

**有 UnieInfra** 時，系統會在這些情況下自動切換到標準 decode，保留 NGRAM 的優勢同時避免效能下降。

### 最新消息

- **2026-01-18**：文檔全面改版 - 解決用戶配置困難的痛點

  **問題背景**：
  - 原文檔結構混亂，資訊分散難以查找
  - 參數說明過於簡略，用戶不清楚如何設定門檻值
  - 缺乏實戰範例，從理論到實踐有斷層
  - 故障排除指引不足，遇到問題難以自行解決

  **改進內容**：
  - ✅ **新增「參數快速參考」**：完整的參數表格，包含 SGLang 基本參數、NGRAM 參數、所有 DSC gate 參數，每個參數都有說明、建議值和使用時機
  - ✅ **擴充「配置指南」**：每個 gate 都有詳細的用途解釋、運作原理、使用時機和範例配置
  - ✅ **新增「預設配置」**：提供「最佳化基線」和「乾淨基線」兩種模板配置
  - ✅ **新增「使用範例」**：從最簡到完整配置的 5 個實戰範例，涵蓋 RAG、程式碼生成、聊天等場景
  - ✅ **新增「理解 Gate 機制」**：深入解釋 NGRAM 為何可能變慢、gate 觸發時機、逐步調整策略
  - ✅ **新增「故障排除」**：4 個常見問題的症狀、原因、解決方案
  - ✅ **擴充 FAQ**：從 4 個增加到 8 個常見問題
  - ✅ **新增「效能建議」**：針對不同工作負載的推薦配置、常見陷阱、監控建議

  **預期效果**：
  - 🎯 新用戶可在 5 分鐘內完成基本配置並運行
  - 🎯 生產用戶能根據工作負載特性選擇最佳配置
  - 🎯 減少因參數設定不當導致的效能問題
  - 🎯 降低技術支援負擔，用戶可自行排除 80% 的常見問題
  - 🎯 中英文文檔完全同步，內容一致性 100%

## 快速開始

### 最簡配置

啟用 NGRAM 並使用基本的 batch size gating：

```bash
python3 -m sglang.launch_server \
  --model <hf_model> \
  --speculative-algorithm NGRAM \
  --speculative-ngram-max-batch-size 15
```

### 預期行為

- **低 batch size（≤15 個請求）**：啟用 NGRAM，加速 token 生成
- **高 batch size（>15 個請求）**：自動回退到標準 decode
- **Gate 狀態**：在伺服器輸出中記錄（查看控制台中的「NGRAM enabled/disabled」訊息）

### 如何驗證運作正常

監控伺服器日誌中的 gate 狀態變更訊息。你應該會看到系統根據當前 batch size 在 NGRAM 和標準 decode 之間切換。

## 預設配置

以下為**模板**配置，請自行填入模型與門檻參數。

### 最佳化基線（Prefill Cache + LMCache + DSC + NGRAM）

適合想要開啟快取與動態切換的預設路徑。

- Prefill cache（RadixAttention prefix cache）**預設開啟**，請不要設 `--disable-radix-cache`。
- LMCache 需要明確開啟。
- DSC + NGRAM 啟用，門檻由使用者填入。

```bash
python3 -m sglang.launch_server \
  --model <hf_model> \
  --speculative-algorithm NGRAM \
  --speculative-ngram-max-batch-size <N> \
  --enable-lmcache
```

可選的 gate（依工作負載需求添加）：

```bash
--speculative-ngram-accept-rate-low <L> \
--speculative-ngram-accept-rate-high <H> \
--speculative-ngram-max-seq-len <M> \
--speculative-ngram-max-new-tokens <K>
```

### 乾淨基線（DSC + NGRAM Only）

UnieInfra 的**最小可執行方案**，不啟用額外快取層。

```bash
python3 -m sglang.launch_server \
  --model <hf_model> \
  --speculative-algorithm NGRAM \
  --speculative-ngram-max-batch-size <N>
```

## 參數快速參考

以下表格列出所有啟動 UnieInfra 時常用的參數。

### 基本參數（SGLang）

| 參數 | 必填 | 說明 | 範例值 |
|------|------|------|--------|
| `--model` | ✅ | HuggingFace 模型路徑或本地路徑 | `meta-llama/Llama-3.1-8B-Instruct` |
| `--host` | ❌ | 伺服器監聽位址 | `127.0.0.1`（預設）或 `0.0.0.0` |
| `--port` | ❌ | 伺服器監聽埠號 | `30000`（預設） |
| `--tp-size` | ❌ | Tensor Parallelism 大小（張量並行） | `1`（預設），GPU 數量 |
| `--mem-fraction-static` | ❌ | 靜態分配的 GPU 記憶體比例 | `0.9`（預設） |

### NGRAM 核心參數

| 參數 | 必填 | 說明 | 範例值 |
|------|------|------|--------|
| `--speculative-algorithm` | ✅ | 啟用推測解碼演算法 | `NGRAM` |
| `--speculative-ngram-capacity` | ❌ | NGRAM cache 容量（token 數） | `10000000`（預設，1000萬） |
| `--speculative-ngram-min-match-window-size` | ❌ | 最小匹配窗口大小 | `1`（預設） |
| `--speculative-ngram-max-match-window-size` | ❌ | 最大匹配窗口大小 | `12`（預設） |
| `--speculative-ngram-branch-length` | ❌ | Draft 分支長度 | `18`（預設） |

### DSC Gate 參數（動態策略控制）

這些參數控制何時啟用/停用 NGRAM。**全部可選**，不設定則該 gate 不生效。

| 參數 | 預設值 | 說明 | 建議值 |
|------|--------|------|--------|
| `--speculative-ngram-max-batch-size` | None | batch size ≤ N 才使用 NGRAM | `10-20` |
| `--speculative-ngram-max-seq-len` | None | 序列長度 < M 才使用 NGRAM | `2048-8192` |
| `--speculative-ngram-max-new-tokens` | None | 輸出長度 ≤ K 才使用 NGRAM | `256-1024` |
| `--speculative-ngram-accept-rate-low` | None | EMA 低於此值時關閉 gate | `0.3-0.5` |
| `--speculative-ngram-accept-rate-high` | None | EMA 高於此值時打開 gate | `0.4-0.6` |
| `--speculative-ngram-accept-rate-ema-decay` | `0.9` | EMA 衰減係數（0-1） | `0.85-0.95` |
| `--speculative-ngram-accept-rate-warmup` | `0` | gate 生效前的樣本數 | `32-128` |
| `--speculative-ngram-accept-rate-probe-interval` | `0` | gate 關閉時的探測間隔（步數） | `10-50` |

### 可選功能參數

| 參數 | 說明 | 何時使用 |
|------|------|----------|
| `--enable-lmcache` | 啟用分層 KV cache | 想增加 cache 容量和命中率 |
| `--disable-radix-cache` | 停用 prefix cache | 一般**不建議**停用 |
| `--enable-metrics` | 啟用 Prometheus 指標 | 需要監控和可觀測性 |
| `--log-requests` | 記錄所有請求詳情 | 除錯或追蹤請求 |

### PD 分離參數（進階）

| 參數 | 說明 | 範例值 |
|------|------|--------|
| `--disaggregation-mode` | PD 分離模式 | `null`（預設）, `prefill`, `decode` |
| `--disaggregation-transfer-backend` | KV transfer 後端 | `mooncake`, `nixl`, `ascend` |

### 參數使用範例

#### 最簡設定（只有 batch gate）
```bash
python3 -m sglang.launch_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --speculative-algorithm NGRAM \
  --speculative-ngram-max-batch-size 15
```

#### 生產環境設定（batch + accept-rate gate）
```bash
python3 -m sglang.launch_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --speculative-algorithm NGRAM \
  --speculative-ngram-max-batch-size 15 \
  --speculative-ngram-accept-rate-low 0.40 \
  --speculative-ngram-accept-rate-high 0.55 \
  --speculative-ngram-accept-rate-warmup 64 \
  --enable-lmcache \
  --enable-metrics
```

#### 完整 gate 設定
```bash
python3 -m sglang.launch_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --speculative-algorithm NGRAM \
  --speculative-ngram-max-batch-size 15 \
  --speculative-ngram-max-seq-len 4096 \
  --speculative-ngram-max-new-tokens 512 \
  --speculative-ngram-accept-rate-low 0.40 \
  --speculative-ngram-accept-rate-high 0.55 \
  --speculative-ngram-accept-rate-ema-decay 0.9 \
  --speculative-ngram-accept-rate-warmup 64 \
  --speculative-ngram-accept-rate-probe-interval 20
```

### 重要提醒

> [!IMPORTANT]
> - **必填參數**：只有 `--model` 和 `--speculative-algorithm NGRAM` 是必須的
> - **Gate 參數都是可選的**：不設定某個 gate 參數，該 gate 就不會限制 NGRAM
> - **從簡單開始**：建議先只用 `--speculative-ngram-max-batch-size`，觀察效果後再加其他 gate
> - **調整門檻**：表格中的「建議值」是起點，需根據實際工作負載調整

## 核心概念

### Prefill vs Decode

- **Prefill**：計算密集型階段，並行處理完整的輸入 prompt
- **Decode**：記憶體密集型階段，逐個 token 生成輸出

UnieInfra **只在 decode 階段運作**。Prefill 行為不受影響。

### NGRAM 推測解碼

NGRAM spec decoding 的運作方式：
1. 在 NGRAM cache 中查找之前見過的 token 序列
2. 使用匹配結果生成多個候選 token（draft）
3. 用目標模型並行驗證這些 draft
4. 接受正確的 token，拒絕錯誤的

**優勢**：當接受率高時，每步可以生成多個 token
**成本**：draft 查找 + 驗證的開銷，目前各階段之間沒有重疊

### 動態策略控制器（DSC）

DSC 在每個 decode step 評估多個「gate」來決定 NGRAM 是否可能有益：

```
當前 batch → 檢查 gates → 使用 NGRAM 或 bypass
```

所有 gate 都是**可選的**。如果你不配置某個 gate，它就不會限制 NGRAM 使用。

### Gate 哲學

Gate 的目的是**防止效能下降**，而非限制良好情境：
- Gate 只會在 NGRAM 可能比標準 decode 更慢時觸發
- 多個 gate 可以組合使用以實現細粒度控制
- 保守的預設值確保安全性優先於最大效能

## 架構

### 元件概覽

- **Scheduler**：編排 batch 處理，擁有 DSC 狀態，做出 gate 決策
- **ScheduleBatch**：容器，包含當前請求、序列長度和 spec info
- **DSC（在 Scheduler 內）**：評估所有 gate 並確定執行路徑
- **NGRAM Worker**：生成 draft、執行驗證、管理 NGRAM cache、支援 bypass
- **Target Worker**：執行實際的模型計算（標準 decode 或驗證）
- **Output Processor**：更新 tokens/logprobs、釋放 KV cache、維護接受率 EMA

### 單機資料流

```
    +----------------------------+
    |        Scheduler Loop      |
    +-------------+--------------+
              |
              v
    +----------------------------+
    |        ScheduleBatch       |
    |    - reqs[]                |
    |    - batch_size()          |
    |    - seq_lens_cpu          |
    +-------------+--------------+
                  |
                  v
    +----------------------------+
    | Dynamic Strategy Controller|
    | (DSC)                      |
    |  gates:                    |
    |   - batch size             |
    |   - accept-rate EMA        |
    |   - max seq len            |
    |   - max_new_tokens         |
    +------+---------------------+
       |                 |
       | use NGRAM       | bypass
       v                 v
+----------------+   +----------------+
| NGRAM Worker   |   | Target Worker  |
| (verify path)  |   | (normal decode)|
+-------+--------+   +--------+-------+
        \                 /
         \               /
          v             v
      +---------------------+
      | Output Processor     |
      | - append output_ids  |
      | - update KV cache    |
      | - update accept EMA  |
      +---------------------+
```

### Decode Step 時序

```
Step 0: 過濾 running batch（移除已完成/撤回的請求）
Step 1: 從可運行的請求建構 decode batch
Step 2: DSC 評估所有 gates → 做出決策（NGRAM 或 bypass）
Step 3: 執行選定的路徑（NGRAM verify 或標準 decode）
Step 4: 處理輸出（追加 tokens、logprobs、釋放 KV）
Step 5: 更新接受率 EMA（僅限 NGRAM steps）
```

### 整合點

DSC 在 scheduler 層級整合到 SGLang：
- 不修改 model executor 或 attention kernel
- 使用現有的 KV cache 管理
- 與所有 sampling backend 兼容
- 使用標準輸出處理流程

## 配置指南

所有 DSC 參數都是可選的。只配置你需要的 gate。

### Gate 1：Batch Size 控制

**參數**：`--speculative-ngram-max-batch-size N`

**目的**：將 NGRAM 限制在效能最佳的小 batch size。

**運作方式**：
- 未設定：batch size 不限制 NGRAM
- N < 1：永遠 bypass NGRAM
- 當前 batch size > N：bypass 到標準 decode
- 當前 batch size ≤ N：允許 NGRAM

**使用時機**：
- 你的工作負載有可變的併發度
- 你知道 NGRAM 在超過某個 batch size 後效益遞減
- 你想要最簡單形式的動態切換

**配置範例**：
```bash
--speculative-ngram-max-batch-size 15
```

**典型值**：大多數工作負載為 10-20，根據硬體和模型調整

### Gate 2：序列長度限制

**參數**：`--speculative-ngram-max-seq-len M`

**目的**：對於記憶體壓力大的長序列停用 NGRAM。

**運作方式**：
- 檢查當前 batch 中的 `max(seq_lens_cpu)`
- 如果最大序列長度 ≥ M：bypass 到標準 decode
- 否則：允許 NGRAM（受其他 gate 限制）

**使用時機**：
- 你服務的請求有高度可變的 context 長度
- 長序列時記憶體頻寬成為瓶頸
- 長 context 的 NGRAM cache 命中率較低

**配置範例**：
```bash
--speculative-ngram-max-seq-len 4096
```

**典型值**：2048-8192，取決於模型 context window 和可用記憶體

### Gate 3：輸出長度限制

**參數**：`--speculative-ngram-max-new-tokens K`

**目的**：對於需要非常長輸出的請求 bypass NGRAM。

**運作方式**：
- 檢查每個請求的 `max_new_tokens` 參數
- 如果任何請求的 max_new_tokens > K：bypass 整個 batch
- 否則：允許 NGRAM（受其他 gate 限制）

**使用時機**：
- 你有混合工作負載（有些短補全，有些長生成）
- 長輸出往往有較低的 NGRAM cache 命中率
- 你想對短輸出優先優化延遲

**配置範例**：
```bash
--speculative-ngram-max-new-tokens 512
```

**典型值**：256-1024，取決於你的典型輸出分佈

### Gate 4：Accept-Rate EMA 搭配遲滯（Hysteresis）

這是最複雜的 gate，使用執行時統計來適應工作負載特性。

#### 核心參數

**低門檻**：`--speculative-ngram-accept-rate-low L`
- 當 EMA < L 時，gate **關閉**（bypass NGRAM）
- 典型值：0.3-0.5

**高門檻**：`--speculative-ngram-accept-rate-high H`
- 當 EMA ≥ H 時，gate **打開**（允許 NGRAM）
- 必須 ≥ L
- 未設定時預設為 L
- 典型值：0.4-0.6

**EMA 衰減**：`--speculative-ngram-accept-rate-ema-decay D`
- 控制對近期樣本的權重
- D = 0.0：只有最新樣本有影響
- D = 1.0：歷史主導
- 預設：0.9
- 典型值：0.85-0.95

**暖機（Warmup）**：`--speculative-ngram-accept-rate-warmup W`
- 應用 gate 前要收集的樣本數
- 防止基於不足數據的過早 gating
- 預設：0（無暖機）
- 典型值：32-128

**探測間隔**：`--speculative-ngram-accept-rate-probe-interval P`
- 當 gate 關閉時，每 P 個 decode step 允許 1 次 NGRAM 步驟
- 重新整理 EMA 以檢測條件是否改善
- P = 0：無探測（gate 保持關閉直到其他 gate 強制重新打開）
- 典型值：10-50

#### Accept-Rate Gating 如何運作

1. **計算**：每次 NGRAM verify step 後：
   ```
   accept_rate = (accepted_tokens + batch_size) / (batch_size * num_draft_tokens)
   EMA = D * EMA_prev + (1 - D) * accept_rate
   ```

2. **遲滯（Hysteresis）**：使用兩個門檻避免震盪：
   ```
   如果 EMA < L  → 關閉 gate（開始 bypassing）
   如果 EMA ≥ H  → 打開 gate（允許 NGRAM）
   如果 L ≤ EMA < H → 保持當前狀態
   ```

3. **暖機**：收集到 W 個樣本前 gate 不生效

4. **探測**：關閉時，定期允許 NGRAM 以重新整理統計

#### 何時使用 Accept-Rate Gating

- **可變工作負載品質**：有些 prompt 命中率高，有些不高
- **未知接受模式**：你事前不知道哪些任務效果好
- **自適應行為**：你希望系統自動學習和適應

#### 配置範例

**保守配置**（優先穩定性）：
```bash
--speculative-ngram-accept-rate-low 0.45 \
--speculative-ngram-accept-rate-high 0.55 \
--speculative-ngram-accept-rate-ema-decay 0.95 \
--speculative-ngram-accept-rate-warmup 64 \
--speculative-ngram-accept-rate-probe-interval 20
```

**激進配置**（最大化 NGRAM 使用）：
```bash
--speculative-ngram-accept-rate-low 0.30 \
--speculative-ngram-accept-rate-high 0.35 \
--speculative-ngram-accept-rate-ema-decay 0.85 \
--speculative-ngram-accept-rate-warmup 32 \
--speculative-ngram-accept-rate-probe-interval 50
```

### 參數總表

| 參數 | 預設值 | 範圍 | 目的 |
|------|--------|------|------|
| `speculative-ngram-max-batch-size` | None | ≥ 0 或 None | 按 batch size 限制 |
| `speculative-ngram-max-seq-len` | None | ≥ 1 或 None | 按最大序列長度限制 |
| `speculative-ngram-max-new-tokens` | None | ≥ 0 或 None | 按最大輸出長度限制 |
| `speculative-ngram-accept-rate-low` | None | 0.0-1.0 或 None | 關閉 gate 的 EMA 門檻 |
| `speculative-ngram-accept-rate-high` | None | 0.0-1.0 或 None | 打開 gate 的 EMA 門檻 |
| `speculative-ngram-accept-rate-ema-decay` | 0.9 | 0.0-1.0 | EMA 歷史權重 |
| `speculative-ngram-accept-rate-warmup` | 0 | ≥ 0 | gate 生效前的樣本數 |
| `speculative-ngram-accept-rate-probe-interval` | 0 | ≥ 0 | 探測間隔步數（0=停用）|

## 使用範例

### 範例 1：僅 Batch Gating（最簡單）

**使用場景**：可變併發度且已知 batch size 門檻

```bash
python3 -m sglang.launch_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --speculative-algorithm NGRAM \
  --speculative-ngram-max-batch-size 15
```

**行為**：
- Batch ≤ 15：啟用 NGRAM
- Batch > 15：標準 decode

**最適合**：簡單部署、可預測的工作負載

### 範例 2：生產環境配置

**使用場景**：具有自適應行為的生產部署

```bash
python3 -m sglang.launch_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --speculative-algorithm NGRAM \
  --speculative-ngram-max-batch-size 15 \
  --speculative-ngram-accept-rate-low 0.40 \
  --speculative-ngram-accept-rate-high 0.55 \
  --speculative-ngram-accept-rate-ema-decay 0.9 \
  --speculative-ngram-accept-rate-warmup 64
```

**行為**：
- 結合 batch size 和 accept-rate gating
- 自動適應工作負載品質
- 暖機防止過早 bypassing

**最適合**：混合工作負載的生產系統

### 範例 3：長 Context 配置

**使用場景**：同時支援短和長 context 請求

```bash
python3 -m sglang.launch_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --speculative-algorithm NGRAM \
  --speculative-ngram-max-batch-size 15 \
  --speculative-ngram-max-seq-len 4096 \
  --speculative-ngram-max-new-tokens 512
```

**行為**：
- NGRAM 僅用於：batch ≤ 15 且 seq_len < 4096 且 output ≤ 512
- 對長 context 或長輸出請求自動 bypass

**最適合**：RAG 系統、可變 context 長度的文件問答

### 範例 4：激進優化

**使用場景**：最大化 NGRAM 使用，願意接受一些效率損失

```bash
python3 -m sglang.launch_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --speculative-algorithm NGRAM \
  --speculative-ngram-max-batch-size 25 \
  --speculative-ngram-accept-rate-low 0.30 \
  --speculative-ngram-accept-rate-high 0.35 \
  --speculative-ngram-accept-rate-ema-decay 0.85 \
  --speculative-ngram-accept-rate-warmup 32 \
  --speculative-ngram-accept-rate-probe-interval 50
```

**行為**：
- 更高的 batch size 門檻
- 更低的 accept-rate 要求
- 更激進的 EMA（對變化反應更快）
- 更頻繁的探測

**最適合**：延遲敏感的應用、已知高品質的工作負載

### 範例 5：保守配置

**使用場景**：穩定性優於效能，避免任何減速風險

```bash
python3 -m sglang.launch_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --speculative-algorithm NGRAM \
  --speculative-ngram-max-batch-size 10 \
  --speculative-ngram-accept-rate-low 0.50 \
  --speculative-ngram-accept-rate-high 0.60 \
  --speculative-ngram-accept-rate-ema-decay 0.95 \
  --speculative-ngram-accept-rate-warmup 128 \
  --speculative-ngram-accept-rate-probe-interval 10
```

**行為**：
- 嚴格的 batch size 限制
- 高 accept-rate 要求
- 慢速 EMA（更穩定，反應較慢）
- 保守的探測

**最適合**：SLA 關鍵的部署、初期測試

## 整合

### PD 分離（Prefill/Decode Disaggregation）

UnieInfra 與 Prefill/Decode 分離完全兼容。在 PD 模式下，**DSC 只在 decode worker 上運行**。Prefill worker 不受影響。

#### PD 架構

```
+-----------+      KV transfer      +-----------+
| Prefill   |  ------------------>  | Decode    |
| Workers   |                       | Workers   |
+-----------+                       +-----+-----+
        \                                 |
         \                                | DSC 決策
          \                               | NGRAM/bypass
           \                        +-----+-----+
            \--------------------> | Router (PD) |
                                   +-------------+
```

#### 單機 PD 設定

```bash
# Prefill worker
python -m sglang.launch_server \
  --model <hf_model> \
  --disaggregation-mode prefill \
  --port 30000

# Decode worker（帶 UnieInfra）
python -m sglang.launch_server \
  --model <hf_model> \
  --disaggregation-mode decode \
  --speculative-algorithm NGRAM \
  --speculative-ngram-max-batch-size 15 \
  --port 30001

# Router
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://127.0.0.1:30000 \
  --decode http://127.0.0.1:30001 \
  --host 0.0.0.0 \
  --port 8000
```

#### Transfer Backend

SGLang 支援多種 KV transfer backend：
- **Mooncake**：高效能 RDMA-based transfer
- **NIXL**：Network-based transfer
- **Ascend**：NPU 專用 backend（用於 Ascend 硬體）
- **Fake**：測試/除錯用 backend

使用 `--disaggregation-transfer-backend <backend>` 指定。

完整的 PD 配置和多機部署請參考 `docs/advanced_features/pd_disaggregation.md`。

#### PD 專屬注意事項

- DSC 決策只應用於 decode worker
- NGRAM cache 維護在 decode worker 上
- Bypass 路徑仍會更新 NGRAM cache（保持可逆性）
- Transfer backend 選擇不影響 DSC 行為

### LMCache 整合

LMCache 是可選的分層 KV cache 層，可以提高 cache 命中率並減少記憶體壓力。

#### LMCache 是什麼？

LMCache 在 GPU KV cache 下方新增次級 cache 層（例如 CPU 記憶體、SSD），實現：
- 更大的有效 cache 容量
- 重複查詢的更好 cache 命中率
- 減少 GPU 記憶體壓力

#### 與 UnieInfra 的關係

LMCache 和 UnieInfra 是**正交的**：
- LMCache 影響 KV 儲存和重用
- UnieInfra 影響 decode 路徑選擇
- 它們可以獨立使用或一起使用

#### 啟用 LMCache

```bash
python3 -m sglang.launch_server \
  --model <hf_model> \
  --speculative-algorithm NGRAM \
  --speculative-ngram-max-batch-size 15 \
  --enable-lmcache
```

#### 配置

LMCache 不會改變 DSC 決策。Gate 仍然根據 batch size、序列長度和接受率運作。

詳細的 LMCache 配置請參考：
- `docs/advanced_features/hicache_design.md`
- `python/sglang/srt/mem_cache/storage/lmcache/README.md`

## 理解 Gate 機制

### 為什麼 NGRAM 可能更慢

NGRAM 推測解碼會增加開銷：

1. **Draft 查找**：在 NGRAM cache 中搜尋匹配
2. **Draft 建構**：建立候選 token 樹
3. **驗證**：對所有候選執行 verify pass
4. **無重疊**：目前實作不會重疊各階段

當接受率高時，每步接受多個 token 可以抵消這些成本。當接受率低或 batch 大時：
- 開銷保持不變
- 收益減少（接受的 token 更少）
- 記憶體頻寬成為瓶頸（大 batch）

結果：標準 decode 可能更快。

### Gate 何時觸發：決策樹

```
開始 decode step
    |
    v
Batch size gate 啟用？
    |
    ├─ 是 → Batch > 門檻？ ─ 是 → BYPASS
    |            |
    |            否
    v            v
序列長度 gate 啟用？
    |
    ├─ 是 → Max seq ≥ 門檻？ ─ 是 → BYPASS
    |            |
    |            否
    v            v
輸出長度 gate 啟用？
    |
    ├─ 是 → 任何輸出 > 門檻？ ─ 是 → BYPASS
    |            |
    |            否
    v            v
Accept-rate gate 啟用？
    |
    ├─ 是 → 暖機完成？
    |            |
    |            ├─ 否 → （允許 NGRAM，收集統計）
    |            |
    |            v
    |         Gate 打開？
    |            |
    |            ├─ 是 → （允許 NGRAM）
    |            |
    |            v
    |         EMA < 低門檻？ ─ 是 → 關閉 gate → BYPASS*
    |            |
    |            否 → EMA ≥ 高門檻？ ─ 是 → 打開 gate
    |                      |
    |                      否 → 保持當前狀態
    v
使用 NGRAM（如果沒有觸發 bypass）

* 有探測時：定期允許 NGRAM 以重新整理 EMA
```

### Gate 交互作用

多個 gate 組合運作：
- **所有 gate 都必須通過**才能運行 NGRAM
- **任一 gate 觸發**都會導致 bypass
- Gate 按順序評估（batch size → seq len → output len → accept-rate）
- 早期 bypass 會跳過後續 gate 評估（輕微優化）

### 調整策略

#### 步驟 1：從簡單開始

僅使用 batch size gating 開始：
```bash
--speculative-ngram-max-batch-size 15
```

#### 步驟 2：監控效能

觀察：
- Token 生成吞吐量
- 延遲（TTFT 和 TPOT）
- GPU/記憶體使用率
- NGRAM 接受率（如果有記錄）

#### 步驟 3：加入 Accept-Rate Gating

如果你看到可變的品質：
```bash
--speculative-ngram-max-batch-size 15 \
--speculative-ngram-accept-rate-low 0.40 \
--speculative-ngram-accept-rate-high 0.50 \
--speculative-ngram-accept-rate-warmup 64
```

#### 步驟 4：調整門檻

- 如果 NGRAM bypass 太頻繁，**降低 accept-rate 門檻**
- 如果你看到減速，**提高 accept-rate 門檻**
- 根據你的硬體和併發模式**調整 batch size**

#### 步驟 5：加入長度 Gate（如需要）

如果你有長 context 工作負載：
```bash
--speculative-ngram-max-seq-len 4096 \
--speculative-ngram-max-new-tokens 512
```

## 常見問題

### Q：為什麼我的 NGRAM 效能在使用 DSC 後變差了？

**A**：DSC 預設是保守的。檢查 gate 是否在不應該的時候 bypass NGRAM：
- 降低你的 accept-rate 門檻
- 提高 batch size 門檻
- 減少暖機期間
- 檢查日誌中的 gate 狀態

### Q：我如何知道 gate 運作正確？

**A**：啟用請求記錄並監控伺服器輸出：
```bash
--log-level debug
```

查找指示 gate 狀態變更和 NGRAM 啟用/停用事件的訊息。

### Q：我應該監控哪些指標？

**A**：關鍵指標：
- **吞吐量**（tokens/秒）：整體系統效能
- **延遲**（TTFT、TPOT）：使用者面向延遲
- **Batch size 分佈**：了解你的工作負載
- **接受率**（如果有公開）：NGRAM 有效性
- **Gate 觸發頻率**：bypass 發生的頻率

### Q：我如何除錯 gate 行為？

**A**：遵循這個檢查清單：
1. 啟用 debug 記錄
2. 檢查當前參數值
3. 即時監控 batch size
4. 追蹤 accept-rate EMA（如需要可加入記錄）
5. 使用合成工作負載測試（控制 batch size）
6. 比較有/無特定 gate 的效能

### Q：何時應該停用某個 gate？

**A**：在以下情況停用 gate：
- 它對你的工作負載觸發太頻繁
- 它沒有提供有意義的保護
- 你想要隔離其效果進行除錯

透過不設定參數或設定極端值（例如 batch size = 1000000）來停用。

### Q：我可以不重啟就改變 gate 參數嗎？

**A**：不行，gate 參數在伺服器啟動時設定。必須重啟伺服器才能改變。

### Q：DSC 與 LoRA adapter 配合嗎？

**A**：是的，DSC 在 scheduler 層級運作，與 LoRA serving 兼容。

### Q：DSC 本身的效能影響是什麼？

**A**：微乎其微。Gate 評估是對 CPU 端 metadata（batch size、序列長度、EMA）的簡單算術運算。開銷相比模型推理可忽略不計。

## 效能建議

### 建議的起始配置

**RAG/問答系統**：
```bash
--speculative-ngram-max-batch-size 15 \
--speculative-ngram-max-seq-len 4096 \
--speculative-ngram-accept-rate-low 0.40 \
--speculative-ngram-accept-rate-high 0.55
```

**程式碼生成**：
```bash
--speculative-ngram-max-batch-size 12 \
--speculative-ngram-max-new-tokens 1024 \
--speculative-ngram-accept-rate-low 0.35 \
--speculative-ngram-accept-rate-high 0.45
```

**聊天/指令遵循**：
```bash
--speculative-ngram-max-batch-size 20 \
--speculative-ngram-accept-rate-low 0.45 \
--speculative-ngram-accept-rate-high 0.60
```

### 常見陷阱

1. **過於激進的門檻**：設定 accept-rate 門檻太高會在有益時也停用 NGRAM
2. **無暖機**：基於不足統計數據的過早 gating
3. **Batch size 太低**：錯過 NGRAM 加速的機會
4. **Batch size 太高**：當標準 decode 更快時仍允許 NGRAM
5. **無探測**：即使條件改善 gate 也保持關閉

### 監控和可觀測性

啟用指標以獲得更好的可見性：
```bash
--enable-metrics \
--log-requests
```

考慮為以下項目加入自定義記錄：
- 當前 batch size
- Gate 觸發事件
- Accept-rate EMA 值
- NGRAM 啟用/停用轉換

## 故障排除

### 問題：NGRAM 從未觸發

**症狀**：伺服器總是使用標準 decode

**可能原因**：
1. Batch size 總是超過門檻
2. Accept-rate gate 卡在關閉狀態
3. 序列/輸出長度 gate 總是觸發

**解決方案**：
- 檢查你的工作負載中的實際 batch size
- 提高 batch size 門檻
- 降低 accept-rate 門檻或停用 accept-rate gate
- 檢視序列/輸出長度分佈

### 問題：效能比沒有 DSC 更差

**症狀**：使用 UnieInfra 時吞吐量降低或延遲增加

**可能原因**：
1. Gate 切換太頻繁（高開銷）
2. 門檻設定不正確
3. 工作負載不適合 NGRAM

**解決方案**：
- 提高 EMA 衰減（更穩定）
- 擴大遲滯間距（high - low）
- 停用有問題的 gate
- 考慮 NGRAM 是否適合你的工作負載

### 問題：Accept-rate gate 保持關閉

**症狀**：Gate 關閉後從不重新打開

**可能原因**：
1. 探測間隔 = 0（無探測）
2. EMA 從未恢復到高門檻
3. 工作負載品質確實很差

**解決方案**：
- 啟用探測：`--speculative-ngram-accept-rate-probe-interval 20`
- 降低高門檻
- 檢查工作負載特性是否改變

### 問題：延遲變異性高

**症狀**：回應時間不一致

**可能原因**：
1. 頻繁的 gate 轉換
2. 工作負載有混合特性
3. EMA 設定太反應靈敏

**解決方案**：
- 提高 EMA 衰減（更平滑的轉換）
- 擴大遲滯間距
- 使用更保守的門檻
- 考慮分離工作負載類型

## 實作細節

### 程式碼位置

核心 DSC 邏輯：
- **Gate 決策邏輯**：`python/sglang/srt/managers/scheduler.py`
  - 方法：`_use_ngram_for_decode()`（或類似）
  - 評估所有 gate 並回傳決策

- **Accept-rate EMA 更新**：`python/sglang/srt/managers/scheduler_output_processor_mixin.py`
  - 在每個 NGRAM verify step 後更新

- **NGRAM worker bypass**：`python/sglang/srt/speculative/ngram_worker.py`
  - 實作 bypass 到標準 decode
  - 即使在 bypass 期間也維護 NGRAM cache

- **CLI 參數**：`python/sglang/srt/server_args.py`
  - 所有 `speculative_ngram_*` 參數
  - 驗證和預設值

### 給貢獻者

如果你要修改 DSC 行為：

1. **Gate 邏輯**：修改 `scheduler.py` 中的 gate 評估
2. **新 gate**：在 `server_args.py` 加入參數，在 scheduler 實作檢查
3. **EMA 計算**：在 output processor mixin 中調整
4. **測試**：使用已知 batch size 和 accept rate 的受控合成工作負載

## 開發路線圖

計劃整合到其他推理 backend：

- [ ] **vLLM**：為 vLLM 的推測解碼實作 DSC
- [ ] **TensorRT-LLM**：支援 TensorRT decode pipeline
- [ ] **LMDeploy**：與 Bytedance LMDeploy 整合
- [ ] **Efficient-Transformers**：支援 Qualcomm AI Cloud

歡迎社群貢獻！

## 優缺點權衡

### 優勢

✅ 在低 batch size 時保留 NGRAM 優勢
✅ 在高併發時保護吞吐量
✅ 自動適應工作負載品質
✅ 切換不需要重啟 worker
✅ 使用現有 decode 邏輯的安全回退
✅ 透過多個 gate 實現細粒度控制

### 限制

⚠️ 需要調整才能獲得最佳效能
⚠️ 收益取決於工作負載接受模式
⚠️ 長序列通常會 bypass NGRAM（收益有限）
⚠️ 非常長的輸出通常會停用 NGRAM
⚠️ Accept-rate gate 是反應性的，非預測性
⚠️ 無內建指標儀表板（需要外部監控）

---

## 授權條款

UnieInfra 文件以非商業用途提供。基礎程式碼授權請參考 [LICENSE](https://github.com/unieai/sglang/blob/main/LICENSE)。

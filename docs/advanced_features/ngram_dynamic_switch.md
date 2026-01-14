# NGRAM 動態切換（依 batch size）

這份說明描述在 `speculative_algorithm=NGRAM` 時，如何依 decode batch size 動態決定是否啟用 NGRAM，以及在 overlap / mixed‑chunk 開啟時的行為。

## 目的與範圍

- 目的：當 batch 小時用 NGRAM 提升小量 decode 的效率；batch 大時改走一般 decode（非 NGRAM）以避免 NGRAM 在大 batch 的成本。
- 範圍：**只影響 decode**（ForwardMode.DECODE）。prefill/extend 不走 NGRAM。
- 前提：server 啟動時仍是 `speculative_algorithm=NGRAM`。

## 重要名詞

- **batch size**：當前 decode batch 的 request 數量（`len(batch.reqs)`），不是 token 數。
- **continue batch**：running decode batch（Scheduler 每一輪 decode 會更新/縮小的那一批）。

## 參數

- `--speculative-ngram-max-batch-size <N>`
  - 未設定：永遠使用 NGRAM。
  - 設定為 N：**當 decode batch size <= N** 時使用 NGRAM，否則走非‑NGRAM。
  - 例：想要「<16 才 NGRAM」，請設 `--speculative-ngram-max-batch-size 15`。

## 什麼時候計算 batch size

- 在 **每一輪 decode forward** 前做判斷。
- 流程簡化如下：
  1. Scheduler 更新 running batch（移除 finished/retracted）。
  2. 取得當輪 decode batch。
  3. NGRAM worker 在 decode forward 前檢查 `batch.batch_size()`。

這表示 **batch size 是動態的**，每一輪都重新判斷。

## 策略切換規則（decode）

假設 `speculative_ngram_max_batch_size = N`：

- **batch size < N**：使用 NGRAM（spec verify）。
- **batch size = N**：仍使用 NGRAM（因為條件是 `<= N`）。
- **batch size > N**：改走非‑NGRAM（bypass NGRAM）。

若你要「<16 才 NGRAM」，實際上就是 `N=15`：

- **1–15**：NGRAM
- **16**：非‑NGRAM
- **>16**：非‑NGRAM

## 各情境行為細節

### 1) batch size < 16（或 <= N）

- 進入 NGRAM spec verify 流程。
- 行為：
  - 產生 draft tokens，切到 `TARGET_VERIFY`。
  - 目標模型 verify，NGRAM 會在 forward 階段更新 `req.output_ids`。
  - logprob 由 NGRAM worker 處理。
  - 完成後回到 `DECODE`。

### 2) batch size = 16

- 是否用 NGRAM **取決於 N**。
  - 若 `N=15`：走非‑NGRAM。
  - 若 `N=16`：仍走 NGRAM。

### 3) batch size > 16（或 > N）

- bypass NGRAM，直接用一般 decode：
  - NGRAM worker 會先準備一個「非 spec decode」的 batch：
    - 取每個 request 的最後一個 token 作為 `input_ids`。
    - 暫時把 `spec_algorithm` 視為 `NONE` 來走標準 decode 的記憶體配置。
  - 走 target worker 的一般 decode。
  - output processor 用非‑spec 路徑處理 logprob 與輸出。
  - decode 產生的下一個 token 會回寫到 NGRAM cache，方便之後再切回 NGRAM。

### 4) continue batch 變小（running batch shrink）

- 每一輪 decode 前都會 `filter_batch()` 移除 finished/retracted。
- 當 batch size 因此降到 <= N 時：**下一輪 decode 會自動回到 NGRAM**。
- 注意：切換是「逐輪」的，不會在同一個 forward 步驟中切換。

## overlap / mixed‑chunk 行為

- NGRAM 現在允許 overlap 與 mixed‑chunk。
- 保護措施：
  - overlap 下若 NGRAM 在 forward 內就把 request 完成，output processor 仍會釋放 KV cache / 記錄 completion time。
  - mixed‑chunk 時，若 running batch 是 NGRAM，會先做非‑spec decode 準備，再 mix 進 prefill batch，避免 input_ids/out_cache_loc 不一致。

## 限制與注意事項

- `speculative_algorithm` 仍是 NGRAM，**整個 server 的 spec 相關設定仍以 NGRAM 為準**（例如 DP attention 仍不支援）。
- CUDA graph / overlap 的行為仍會受到 NGRAM 啟動時的捕捉模式影響，當 bypass NGRAM 時可能不會得到完全等同「原本非‑NGRAM」的效能。
- 這個動態切換 **只影響 decode**；prefill/extend 不會走 NGRAM。

## 範例

```
python3 -m sglang.launch_server \
  --model <hf_model> \
  --speculative-algorithm NGRAM \
  --speculative-ngram-max-batch-size 15
```

行為：
- decode batch size 1–15 → NGRAM
- decode batch size >= 16 → 非‑NGRAM

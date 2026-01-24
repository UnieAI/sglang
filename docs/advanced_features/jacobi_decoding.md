# Jacobi Decoding in SGLang

Jacobi Decoding (or Lookahead Decoding) is a parallel decoding algorithm that accelerates inference by predicting and verifying multiple tokens in a single forward pass. Unlike NGRAM, it relies on the model's own capabilities (and optionally an n-gram pool of recent tokens) rather than a strict historical cache match.

## Key Features

- **Self-Speculative**: Uses the target model itself for verification, requiring no external draft model.
- **Prefix-Independent**: Can accelerate generation even without exact historical matches (though benefits from local repetition via the n-gram pool).
- **Dynamic Strategy Control (DSC)**: Automatically enables/disables optimization based on runtime conditions.
- **Fully Compatible**: Works with LMCache and PD Disaggregation.

## Usage

Enable Jacobi Decoding by setting `--speculative-algorithm JACOBI`.

### Basic Usage

```bash
python -m sglang.launch_server \
  --model <your-model> \
  --speculative-algorithm JACOBI \
  --jacobi-steps-per-yield 5 \
  --jacobi-ngram-pool-size 20
```

### Dynamic Strategy Control (DSC)

To prevent performance degradation in adverse scenarios (e.g., high batch size or low acceptance rates), use the following gating arguments:

- `--jacobi-max-batch-size <N>`: Disable Jacobi when batch size > N.
- `--jacobi-accept-rate-low <L>`: Disable Jacobi if acceptance rate drops below L (e.g. 0.3).
- `--jacobi-accept-rate-high <H>`: Re-enable Jacobi if acceptance rate rises above H (e.g. 0.4).
- `--jacobi-accept-rate-probe-interval <P>`: When disabled, try enabling Jacobi every P steps to check if conditions improved.

### LMCache & PD Integration

Jacobi Decoding is designed to be orthogonal to memory management:
- **LMCache**: It utilizes the standard ModelRunner, so LMCache's KV offloading and retrieval work seamlessly.
- **PD Separation**: The "Decode" worker handles the Jacobi logic independently. If DSC triggers a fallback, it simply behaves as a standard decode step, maintaining full compatibility with the Prefill-Decode architecture.

## Comparison: Jacobi vs NGRAM

| Feature | NGRAM | Jacobi |
|---|---|---|
| **Mechanism** | Matches exact history sequences | Model-based lookahead (iteration) |
| **Best For** | Code, Logs, Repetitive Structured Text | General Text, Creative Writing |
| **Throughput** | 5-10x (on hit) | 1.5-3x (more consistent) |
| **Overhead** | Low (CPU lookup) | Medium (Model Forward) |

For most general-purpose serving where user queries vary, Jacobi offers a robust middle ground.

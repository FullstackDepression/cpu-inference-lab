# cpu-inference-lab

How fast can a language model run on a 2-core Celeron with no GPU — and *why*
exactly that fast. Every number here comes from a run on one machine, with the
command to reproduce it.

**Machine:** Intel Celeron 6305 (2 cores, 1.8 GHz), 16 GB DDR4, Debian 13,
llama.cpp `23f7aab`, CPU only.

## Finding 1 — speed follows weights touched per token, not file size

A rule of thumb: generation reads all weights once per token, so
`tokens/s ≈ memory bandwidth ÷ model size`. Measured bandwidth here is ~10 GB/s
(1.28 GB × 8 tok/s). It predicted Qwen3.5-4B (2.74 GB) at 3.6 tok/s — measured 3.4.

Then it broke: gemma-4-E2B is 2.4× larger than Qwen3.5-2B and generates just as fast.

| Model | File | tok/s | cycles | instructions | IPC | cache misses |
|---|---|---|---|---|---|---|
| Qwen3.5-2B Q4_K_M | 1.18 GiB | 7.43 ± 1.81 | 126.1 B | 276.5 B | 2.19 | 4.15 B |
| gemma-4-E2B Q4_K_M | 2.88 GiB | 7.51 ± 0.64 | 132.3 B | 322.0 B | 2.43 | 4.13 B |

Raw data: [`results/2026-11-24-perf-stat.csv`](results/2026-11-24-perf-stat.csv).

**Why.** Both models miss the cache the same number of times (4.1 B), so per token
they pull about the same amount of data from RAM — even though one file is 2.4×
bigger. `bench/tensors.py` shows where the extra size goes:

```
gemma-4-E2B:  1.50 GiB  52%  per_layer_token_embd.weight
Qwen3.5-2B:   0.39 GiB  33%  token_embd.weight
```

Half of gemma's file is a per-layer embedding table. It is a lookup: each token
reads one row of it, not the whole table. Take it out and gemma touches ~1.4 GiB
per token — about the same as Qwen 2B. Same work, same speed.

So the rule of thumb is right only if "model size" means **bytes touched per
token**, not the size of the file on disk.

**What it does *not* show.** IPC is 2.2–2.4: the cores are busy executing, not
mostly stalled on memory. Both the memory traffic and the compute (dequantizing
Q4_K blocks and multiplying) scale with weights touched, so this result alone
can't say which of the two is the limit. Finding 2 looks at that.

### Reproduce

```bash
sudo bench/perf-stat.sh models/Qwen3.5-2B-Q4_K_M.gguf
sudo bench/perf-stat.sh models/gemma-4-E2B-it-Q4_K_M.gguf
python3 bench/tensors.py models/gemma-4-E2B-it-Q4_K_M.gguf
```

## Finding 2 — the second core helps prompt processing more than generation

Same model (Qwen3.5-2B Q4_K_M), 128-token prompt, 64 generated tokens, 3 repetitions.

| Threads | Prompt processing, tok/s | Generation, tok/s |
|---|---|---|
| 1 | 24.04 ± 0.20 | 6.02 ± 0.05 |
| 2 | 37.11 ± 4.30 | 7.85 ± 0.55 |
| gain | **×1.54** | **×1.30** |

Raw data: [`results/2026-11-24-threads.csv`](results/2026-11-24-threads.csv).

Prompt processing handles all 128 tokens in one batch: every weight loaded from
RAM is reused 128 times, so it is mostly arithmetic and a second core helps a lot.
Generation handles one token at a time: every weight is loaded and used once, so
there is far less arithmetic per byte and the second core helps less — part of the
time both cores wait on the same memory.

Effective rate in generation: 1.27 GB × 7.85 tok/s ≈ 10 GB/s, well below what
DDR4 promises on paper. Whether that is the memory controller, the two cores'
ability to issue loads, or the dequantization work is the next question.

Reproduce: `llama-bench -m <model> -t 1 -p 128 -n 64 -r 3`, then `-t 2`.

## Next

- `perf stat` on prompt processing vs generation: compare IPC and cache misses per token.
- Check the real memory configuration (channels, speed) and measure bandwidth with a plain streaming test.
- `perf record` on generation: which kernel functions take the time.

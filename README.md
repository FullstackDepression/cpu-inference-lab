# llm-perf-lab

How fast can a language model run on a 2-core Celeron with no GPU — and *why*
exactly that fast. Every number here comes from a run on one machine, with the
command to reproduce it.

**Machine:** Intel Celeron 6305 (2 cores, 1.8 GHz), 16 GB DDR4, Debian 13,
llama.cpp `23f7aab`, CPU only.

## Finding 1 — speed follows bytes read per token, not file size

A rule of thumb: generation reads all weights once per token, so
`tokens/s ≈ memory bandwidth ÷ model size`. Measured bandwidth here is ~10 GB/s
(1.28 GB × 8 tok/s). It predicted Qwen3.5-4B (2.74 GB) at 3.6 tok/s — measured 3.4.

Then it broke: gemma-4-E2B is 2.4× larger than Qwen3.5-2B and generates just as fast.

| Model | File | tok/s | cycles | instructions | IPC | cache misses |
|---|---|---|---|---|---|---|
| Qwen3.5-2B Q4_K_M | 1.18 GiB | 7.43 ± 1.81 | 126.1 B | 276.5 B | 2.19 | 4.15 B |
| gemma-4-E2B Q4_K_M | 2.88 GiB | 7.51 ± 0.64 | 132.3 B | 322.0 B | 2.43 | 4.13 B |

Raw data: [`results/2026-11-24-perf-stat.csv`](results/2026-11-24-perf-stat.csv).

<!-- ROMA: why. Write in your own words, 5–8 lines: what the equal cache-miss
     count tells you, what bench/tensors.py showed about gemma's file, and how the
     rule of thumb has to change. -->

### Reproduce

```bash
sudo bench/perf-stat.sh ~/llm-models/Qwen3.5-2B-Q4_K_M.gguf
sudo bench/perf-stat.sh ~/llm-models/gemma-4-E2B-it-Q4_K_M.gguf
python3 bench/tensors.py ~/llm-models/gemma-4-E2B-it-Q4_K_M.gguf
```

## Next

- Threads 1 → 2: does the second core help if the bottleneck is memory?
- Prompt processing vs generation: one is compute-bound, the other memory-bound — show it with IPC.
- `perf record` on generation: which kernel functions take the time.

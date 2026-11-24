#!/usr/bin/env bash
# Generation benchmark of one GGUF model under perf stat.
# Usage: sudo bench/perf-stat.sh <model.gguf> [threads] [tokens]
set -euo pipefail
model=${1:?model.gguf}; threads=${2:-2}; tokens=${3:-64}
bench=${LLAMA_BENCH:-$HOME/src/llama.cpp/build/bin/llama-bench}

perf stat -x, -e cycles,instructions,cache-misses -o /tmp/perf.$$ \
  "$bench" -m "$model" -t "$threads" -p 0 -n "$tokens" -o csv > /tmp/bench.$$

tg=$(tail -1 /tmp/bench.$$ | awk -F, '{print $(NF-1)}' | tr -d '"')
get() { awk -F, -v e="$1" '$3 ~ e {print $1}' /tmp/perf.$$; }
echo "model,threads,tg_tok_s,cycles,instructions,cache_misses"
echo "$(basename "$model"),$threads,$tg,$(get cycles),$(get instructions),$(get cache-misses)"
rm -f /tmp/perf.$$ /tmp/bench.$$

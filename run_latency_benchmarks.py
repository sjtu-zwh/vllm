#!/usr/bin/env python3
"""
Script to run benchmark_latency.py with various configurations.
Runs benchmarks with:
- batch_size: [1, 2, 4, 8, 16, 24, 32]
- input_len: [128, 256, 512, 1024]
- output_len: [1024]
"""

import os
import subprocess
import time
from pathlib import Path

# Parameters to test
BATCH_SIZES = [24]
INPUT_LENS = [1]
OUTPUT_LEN = 1024

# Create results directory
RESULTS_DIR = Path("benchmark_results") / f"latency_benchmark_{int(time.time())}"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
print(f"Results will be saved to: {RESULTS_DIR}")

# Model to use - you can change this as needed
MODEL = "/state/partition/whzhang/llama-3.1-8b-instruct"

# Number of iterations
NUM_ITERS_WARMUP = 1
NUM_ITERS = 1

# Run benchmarks for all combinations
for batch_size in BATCH_SIZES:
    for input_len in INPUT_LENS:
        print(f"\n{'='*80}")
        print(f"Running benchmark with batch_size={batch_size}, input_len={input_len}, output_len={OUTPUT_LEN}")
        print(f"{'='*80}")
        
        # Create a descriptive filename for the results
        result_file = RESULTS_DIR / f"result_bs{batch_size}_in{input_len}_out{OUTPUT_LEN}.json"
        
        # Build the command
        cmd = [
            "python", "benchmarks/benchmark_latency.py",
            "--model", MODEL,
            "--batch-size", str(batch_size),
            "--input-len", str(input_len),
            "--output-len", str(OUTPUT_LEN),
            "--num-iters-warmup", str(NUM_ITERS_WARMUP),
            "--num-iters", str(NUM_ITERS),
            "--output-json", str(result_file),
            "--max-num-seqs", "32",
            "--no-enable-prefix-caching",
            "--disable-log-stats",
            "--gpu-memory-utilization", "0.9",
            "--max-model-len", "2048",
            "--max-loras", "4",
            "--max-lora-rank", "8",
            "--lora-modules", "lora0=Nutanix/Meta-Llama-3-8B-Instruct_lora_8_alpha_16",
            "--enable-lora"
        ]
        
        # Run the benchmark
        try:
            subprocess.run(cmd, check=True)
            print(f"Results saved to {result_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error running benchmark: {e}")
            
print("\nAll benchmarks completed!")
print(f"Results are saved in: {RESULTS_DIR}")

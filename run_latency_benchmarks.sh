#!/bin/bash
# Script to run benchmark_latency.py with various configurations
# Runs benchmarks with:
# - batch_size: [1, 2, 4, 8, 16, 24, 32]
# - input_len: [128, 256, 512, 1024]
# - output_len: [1024]

# Create results directory
TIMESTAMP=$(date +%s)
RESULTS_DIR="benchmark_results/latency_benchmark_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"
echo "Results will be saved to: $RESULTS_DIR"

# Model to use - you can change this as needed
MODEL="facebook/opt-125m"  # Using a small model for quick testing

# Number of iterations
NUM_ITERS_WARMUP=3
NUM_ITERS=10

# Run benchmarks for all combinations
for BATCH_SIZE in 1 2 4 8 16 24 32; do
  for INPUT_LEN in 128 256 512 1024; do
    OUTPUT_LEN=1024
    
    echo -e "\n================================================================================"
    echo "Running benchmark with batch_size=$BATCH_SIZE, input_len=$INPUT_LEN, output_len=$OUTPUT_LEN"
    echo "================================================================================"
    
    # Create a descriptive filename for the results
    RESULT_FILE="${RESULTS_DIR}/result_bs${BATCH_SIZE}_in${INPUT_LEN}_out${OUTPUT_LEN}.json"
    
    # Run the benchmark
    python benchmarks/benchmark_latency.py \
      --model "$MODEL" \
      --batch-size "$BATCH_SIZE" \
      --input-len "$INPUT_LEN" \
      --output-len "$OUTPUT_LEN" \
      --num-iters-warmup "$NUM_ITERS_WARMUP" \
      --num-iters "$NUM_ITERS" \
      --output-json "$RESULT_FILE"
      
    if [ $? -eq 0 ]; then
      echo "Results saved to $RESULT_FILE"
    else
      echo "Error running benchmark"
    fi
  done
done

echo -e "\nAll benchmarks completed!"
echo "Results are saved in: $RESULTS_DIR"

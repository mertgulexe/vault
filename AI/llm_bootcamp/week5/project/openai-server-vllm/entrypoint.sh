#!/bin/bash

# Default values
MODEL=${MODEL:-"meta-llama/Meta-Llama-3-8B-Instruct"}
DTYPE=${DTYPE:-"half"}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-2048}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.99}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-1024}
QUANTIZATION=${QUANTIZATION:-"bitsandbytes"}
LOAD_FORMAT=${LOAD_FORMAT:-"bitsandbytes"}
ENFORCE_EAGER=${ENFORCE_EAGER:-true}

# Check and set permissions for directories
for dir in /tmp/huggingface /tmp/cache /tmp/numba_cache /tmp/outlines_cache /tmp/config; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
    fi
    chmod -R 777 "$dir"
    echo "Permissions for $dir:"
    ls -la "$dir"
done

# Construct the command
CMD="vllm serve $MODEL \
--host 0.0.0.0 \
--port 8000 \
--dtype $DTYPE \
--max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
--max-num-seqs $MAX_NUM_SEQS \
--gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
--max-model-len $MAX_MODEL_LEN \
--quantization $QUANTIZATION \
--load-format $LOAD_FORMAT"

# Add enforce-eager only if it's set to true
if [ "$ENFORCE_EAGER" = "true" ]; then
    CMD="$CMD --enforce-eager"
fi

# Execute the command
echo "Running command: $CMD"
exec $CMD
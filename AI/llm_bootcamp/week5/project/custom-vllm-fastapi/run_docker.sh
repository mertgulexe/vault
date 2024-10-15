docker run \
    --runtime "nvidia" \
    --gpus "all" \
    -v ~/.cache/huggingface:/home/user/.cache/huggingface \
    --env-file "../.env" \
    -p 8000:8000 \
    --ipc=host \
    "custom-vllm-fastapi:latest"
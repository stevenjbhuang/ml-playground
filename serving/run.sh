#!/bin/bash

# Start vllm server
vllm serve cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit \
    --quantization compressed-tensors \
    --tensor-parallel-size 2 \
    --max-model-len 65536 \
    --reasoning-parser qwen3
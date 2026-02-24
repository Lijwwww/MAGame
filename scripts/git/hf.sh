#!/bin/bash

# 2. 上传测试数据到 HF Dataset
echo "Uploading results to HF..."
huggingface-cli upload jvvww/MAGame ./results/eval /results/eval --repo-type=dataset --include="*.npy"
huggingface-cli upload jvvww/MAGame ./results/eval_multi_ckps /results/eval_multi_ckps --repo-type=dataset --include="*.npy"

# 3. 上传权重到 HF Model
echo "Uploading checkpoints to HF..."
huggingface-cli upload jvvww/MAGame ./checkpoints /checkpoints --repo-type=dataset

echo "All backups completed!"
#!/bin/bash
TOKEN_FILE="scripts/backup/.tokens.env"
source "$TOKEN_FILE"

# 2. 上传测试数据到 HF Dataset
echo "Uploading results to HF..."
HF_ENDPOINT=https://hf-mirror.com HF_TOKEN=$HF_TOKEN hf upload jvvww/MAGame ./results/eval /results/eval --repo-type=dataset --include="*.npy"
HF_ENDPOINT=https://hf-mirror.com HF_TOKEN=$HF_TOKEN hf upload jvvww/MAGame ./results/eval_multi_ckps /results/eval_multi_ckps --repo-type=dataset --include="*.npy"

# 3. 上传权重到 HF Model
echo "Uploading checkpoints to HF..."
HF_ENDPOINT=https://hf-mirror.com HF_TOKEN=$HF_TOKEN hf upload jvvww/MAGame ./checkpoints /checkpoints --repo-type=dataset

echo "All backups completed!"

# HF_ENDPOINT=https://hf-mirror.com HF_TOKEN=$HF_TOKEN hf upload jvvww/MAGame ~/下载/pursuit-sim.tar pursuit-sim.tar --repo-type=dataset

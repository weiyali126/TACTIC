#!/bin/bash

logdir="./logs"
predictPort=20063
vllmModelPort=8000

echo "starting server..."

# # Deploy vLLM model server locally (optional)
# echo "Starting vLLM server..."
# export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# nohup vllm serve Qwen/Qwen3-30B-A3B --port $vllmModelPort --served-model-name Qwen3-30B-A3B --trust-remote-code --tensor-parapple-size 2 --max-model-len 8192 --gpu-memory-utilization 0.85 > $logdir/vllm.log 2>&1 &

# # Check vllm server
# echo "Waiting for vLLM server to be ready, checking vllm log..."
# total_wait=0
# max_wait=600
# while true; do
#     if grep -q "Starting vLLM API server on http://0.0.0.0:$vllmModelPort" $logdir/vllm.log; then
#         echo "vLLM server is ready!"
#         sleep 10
#         break
#     elif [ $total_wait -ge $max_wait ]; then
#         echo "Max time reached, vLLM server is ready!"
#         sleep 10
#         break
#     else
#         echo "Waiting for vLLM server, sleeping 10s..."
#         total_wait=$((total_wait + 10))
#         sleep 10
#     fi
# done

# start backend server
nohup uvicorn tactic.app.server:app --host "0.0.0.0" --port $predictPort >> $logdir/server.log 2>&1 &
sleep 10
echo "start backend server complete!"

# start frontend server
nohup python -m tactic.app.frontend >> $logdir/frontend.log 2>&1 &
sleep 10
echo "start frontend server complete!"

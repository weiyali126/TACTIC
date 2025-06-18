#!/bin/bash

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
logdir="./logs"
predictPort=20063
modelName="Qwen3-32B"
modelPath="Qwen/Qwen3-32B"
modelPort=8000
gpus="0,1"

echo "starting server..."

# # Deploy vLLM model server locally (optional)
# echo "Starting vLLM server..."
# CUDA_VISIBLE_DEVICES=${gpus} nohup vllm serve ${modelPath} --port ${modelPort} --served-model-name ${modelName} --trust-remote-code --tensor-parallel-size 2 --max-model-len 8192 --gpu-memory-utilization 0.85 --chat-template ${modelPath}/qwen3_nothinking.jinja > ${logdir}/vllm.log 2>&1 &

# # Check vllm server
# echo "Waiting for vLLM server to be ready, checking vllm log..."
# total_wait=0
# max_wait=600
# while true; do
#     if grep -q "Starting vLLM API server on http://0.0.0.0:${modelPort}" ${logdir}/vllm.log; then
#         echo "vLLM server is ready!"
#         sleep 10
#         break
#     elif [ ${total_wait} -ge ${max_wait} ]; then
#         echo "Max time reached, please check vllm.log!"
#         sleep 10
#         break
#     else
#         echo "Waiting for vLLM server, sleeping 10s..."
#         total_wait=$((total_wait + 10))
#         sleep 10
#     fi
# done

# start backend server
nohup uvicorn tactic.app.server:app --host "0.0.0.0" --port ${predictPort} >> ${logdir}/server.log 2>&1 &
sleep 10
echo "start backend server complete!"

# start frontend server
nohup python -m tactic.app.frontend >> ${logdir}/frontend.log 2>&1 &
sleep 10
echo "start frontend server complete!"

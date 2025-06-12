import os
import sys
import time
import requests
import subprocess
from loguru import logger
from vllm import LLM, SamplingParams
from dotenv import load_dotenv, find_dotenv 

load_dotenv(find_dotenv())


def start_vllm_server(vllm_engine_args: dict, timeout: int = 3600) -> None:
    """
    Start the vLLM server and wait for available.
    """
    port = vllm_engine_args.get("port", 8000)
    model = vllm_engine_args.get("model", "")
    served_model_name = vllm_engine_args.get("served_model_name", "")
    tensor_parallel_size = vllm_engine_args.get("tensor_parallel_size", 1)
    gpu_memory_utilization = vllm_engine_args.get("gpu_memory_utilization", 0.9)
    max_model_len = vllm_engine_args.get("max_model_len", 8192)
    trust_remote_code = vllm_engine_args.get("trust_remote_code", True)
    gpus = vllm_engine_args.get("gpus", "0,1,2,3,4,5,6,7")
    enable_thinking = vllm_engine_args.get("enable_thinking", "True")
    
    cmd = [
        "vllm", "serve", model,
        "--port", str(port),
        "--served-model-name", served_model_name,
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
    ]

    if trust_remote_code:
        cmd.append("--trust-remote-code")

    # download qwen3 nothinking chat-template
    if served_model_name.startswith('Qwen3') and not enable_thinking:
        cmd.extend(["--chat-template", "./qwen3_nonthinking.jinja"])

    logger.info(f"Starting {served_model_name} vLLM server on http://0.0.0.0:{port}/v1...")

    # Copy current environment, add CUDA_VISIBLE_DEVICES
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus

    try:
        vllm_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=False,
            env=env,
        )

        # Listen to the log & poll API to make sure the server is available
        check_vllm_server(vllm_process, served_model_name, port, timeout)

        return vllm_process

    except Exception as e:
        logger.error(f"Failed to start vLLM server for {served_model_name}: {e}")
        raise

def check_vllm_server(process: subprocess.Popen, served_model_name: str, port: int, timeout: int = 180):
    """
    Listen for vLLM server logs and poll API endpoints.
    """
    start_time = time.time()
    api_url = f"http://0.0.0.0:{port}/v1/models"

    while True:
        try:
            # Check subprocess status
            if process.poll() is not None:
                logger.info(f"Error: {served_model_name} vLLM server crashed.")
                raise RuntimeError(f"{served_model_name} vLLM server exited unexpectedly.")

            # Output subprocess log
            line = process.stdout.readline()
            if line:
                sys.stdout.write(line)
                sys.stdout.flush()

            # Poll API to check for successful startup
            response = requests.get(api_url, timeout=3)
            if response.status_code == 200:
                logger.info(f"\n{served_model_name} vLLM server is ready.")
                break

        except requests.ConnectionError:
            elapsed_time = time.time() - start_time
            time.sleep(2)

        # Timeout detection
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            logger.error(f"Error: {served_model_name} vLLM server failed to start within {timeout} seconds.")
            process.terminate()
            raise RuntimeError(f"{served_model_name} vLLM server startup timeout.")

    # Stop log output after successful startup 
    process.stdout.close()


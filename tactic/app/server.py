# -*- coding: utf-8 -*-
import os
import json
import time
import uuid
import asyncio
import requests
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from quart import Quart, request, jsonify, Response
from loguru import logger
from dotenv import load_dotenv, find_dotenv 
from tactic.utils import setup_logger, LANG_TABLE
from tactic.app.methods import run_tactic_light, run_tactic_light_with_stream

app = Quart(__name__)

# Set max concurrency, backend server port
MAX_CONCURRENT_REQUESTS = 5
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
BACKEND_SERVER_PORT = 20063

# Set the model information
model_platform = "DeepSeek"
model_name = "deepseek-chat"

# Setup log
configs = {'log_dir': 'logs'}
setup_logger(configs)

def generate_request_id():
    '''Generate a unique request ID'''
    return str(uuid.uuid4())

def parse_messages(messages):
    '''
    Initialize and parse messages, parsing only in a single round
    '''
    system_message, user_message = None, None
    for message in messages:
        role = message.get('role')
        content = message.get('content')
        if role == "system":
            system_message = content
        elif role == "user":
            user_message = content
    # If the user message is not found, an exception will be thrown
    if user_message is None:
        raise ValueError("Request must contain a 'user' message.")

    return system_message, user_message

class Message(BaseModel):
    role: str
    content: str

class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class OpenAIResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage

def construct_response(
    request_id: str,
    model_name: str,
    system_message: Optional[str],
    user_message: str,
    translation: str,
    tokenizer=None
) -> dict:
    # Splicing Prompt (OpenAI format)
    full_prompt = ""
    if system_message:
        full_prompt += f"<|system|>\n{system_message}\n"
    full_prompt += f"<|user|>\n{user_message}\n"

    # Token count
    if tokenizer is not None:
        prompt_tokens = len(tokenizer.encode(full_prompt, add_special_tokens=False))
        completion_tokens = len(tokenizer.encode(translation, add_special_tokens=False))
        total_tokens = prompt_tokens + completion_tokens
    else:
        prompt_tokens = completion_tokens = total_tokens = 0

    # tectonic response
    response = OpenAIResponse(
        id=f"chatcmpl-{request_id}",
        created=int(time.time()),
        model=model_name,
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content=translation),
                finish_reason="stop"
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens
        )
    )
    return response.dict()

async def limited_concurrency(func, *args, **kwargs):
    async with semaphore:
        return await func(*args, **kwargs)

async def limited_concurrency_stream(func, *args, **kwargs):
    async with semaphore:
        async for chunk in func(*args, **kwargs):
            yield chunk

async def handle_abnormal_input(request_id: str, request_model_name: str, sys_msg: str, user_msg: str, stream: bool):
    """
    Checks if user messages are empty and returns a 200 OK response with empty translation.
    """
    if not user_msg:
        if stream:
            async def empty_sse_response():
                yield 'data: [DONE]\n\n'
            return Response(empty_sse_response(), content_type="text/event-stream")
        else:
            response = construct_response(request_id, request_model_name, sys_msg, user_msg, "")
            return jsonify(response), 200

    return None # Return None if input is not empty, signaling the main route to continue

@app.route('/v1/chat/completions', methods=['POST'])
async def chat_completions():
    '''
    OpenAI Chat Completions API-compatible endpoint
    '''
    request_id = generate_request_id()
    data = await request.get_json()
    logger.info(f"Request ID: {request_id} - Request parameters: {data}")
    
    try:
        # Get core parameters
        request_model_name = data.get('model', model_name)  
        messages = data.get('messages', [])
        temperature = data.get('temperature', 0.6)
        max_tokens = data.get('max_tokens', 2048)
        stream = data.get('stream', False)
        
        # Extract prompt
        sys_msg, full_user_msg = parse_messages(messages)
        language_pair = full_user_msg.split(':\n', 1)[0].strip()
        user_msg = full_user_msg.split(':\n', 1)[1].strip()
        src_lang, tgt_lang = language_pair.split('-')
        src_fullname, tgt_fullname = LANG_TABLE[src_lang], LANG_TABLE[tgt_lang]
        source_text = user_msg

        # Handle abnormal input
        abnormal_input_response = await handle_abnormal_input(request_id, request_model_name, sys_msg, user_msg, stream)
        if abnormal_input_response:
            return abnormal_input_response
            
        if stream:
            async def sse_response():
                try:
                    async for chunk in limited_concurrency_stream(run_tactic_light_with_stream, 
                        source_text, src_lang, tgt_lang, src_fullname, tgt_fullname,
                        model_platform, model_name, temperature, max_tokens, request_id
                    ):
                        yield chunk
                except Exception as e:
                    logger.error(f"[{request_id}] Stream error: {str(e)}")
                    yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
                    yield "data: [DONE]\n\n"

            headers = {
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Content-Type': 'text/event-stream',
                'X-Accel-Buffering': 'no'  
            }
                    
            return Response(sse_response(), content_type="text/event-stream; charset=utf-8", headers=headers)
            
        # Non-stream
        translation = await limited_concurrency(run_tactic_light,
            source_text, src_lang, tgt_lang, src_fullname, tgt_fullname,
            model_platform, model_name, temperature, max_tokens, request_id
        )
        response = construct_response(request_id, request_model_name, sys_msg, user_msg, translation)
        logger.info(f"Request ID: {request_id} - Response: {response}")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Request ID: {request_id} - Error: {str(e)}")
        return jsonify({"error": str(e), "request_id": request_id}), 500
    
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=BACKEND_SERVER_PORT)

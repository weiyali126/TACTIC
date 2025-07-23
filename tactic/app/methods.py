# -*- coding: utf-8 -*-
import os
import json
import time
import datetime
import uuid
import asyncio
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from loguru import logger
from dotenv import load_dotenv, find_dotenv
from tactic.agents.draft_agent import DraftAgent
from tactic.agents.refinement_agent import RefinementAgent


async def run_tactic_light(
    source_text: str,
    src_lang: str,
    tgt_lang: str,
    src_fullname: str,
    tgt_fullname: str,
    model_platform: str,
    model_name: str,
    temperature: float=0.6,
    max_tokens: int=2048,
    request_id: str=None
):
    '''
    tactic-light translation
    '''
    start_time = time.time()
    logger.info(f"Source Text | {source_text}")

    # DraftAgent
    draft_agent = DraftAgent(src_fullname, tgt_fullname, model_platform, model_name, temperature, max_tokens)
    draft_translations = await draft_agent.run_common(source_text, src_fullname, tgt_fullname)
    logger.info(f"Elapsed Time: {int(time.time() - start_time)} s. | DraftAgent ｜ {draft_translations}")
    candidate_translations = [draft_translations]

    # RefinementAgent
    refinement_agent = RefinementAgent(src_fullname, tgt_fullname, model_platform, model_name, temperature, max_tokens)
    translation, response = await refinement_agent.run_common(
        src_fullname,
        tgt_fullname,
        source_text,
        candidate_translations,
    )
    logger.info(f"Elapsed Time: {int(time.time() - start_time)} s. | RefinementAgent ｜ {response}")
    logger.info(f"Elapsed Time: {int(time.time() - start_time)} s. | Final Translation ｜ {translation}")

    return translation

def event(content: str, request_id=str(uuid.uuid4()), model_name=None, event_type=None, is_done=False):
    """
    Construct a JSON data volume that conforms to the streaming response format of the OpenAI/DeepSeek API.
    """
    return json.dumps({
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "delta": {"content": content},
            "index": 0,
            "finish_reason": "stop" if is_done else None
        }],
        "type": event_type
    }, ensure_ascii=False)

async def await_while_task_running_with_heartbeat(task, request_id, model_name, interval=10.0):
    last_heartbeat_time = time.time()
    while not task.done():
        now = time.time()
        if now - last_heartbeat_time > interval:
            content = f"[heartbeat] {datetime.datetime.utcnow().isoformat()}"
            yield f"data: {event(content, request_id, model_name, 'Heartbeat')}\n\n"
            last_heartbeat_time = now
        await asyncio.sleep(0.1)

async def run_tactic_light_with_stream(
    source_text: str,
    src_lang: str,
    tgt_lang: str,
    src_fullname: str,
    tgt_fullname: str,
    model_platform: str,
    model_name: str,
    temperature: float = 0.6,
    max_tokens: int = 2048,
    request_id: str = None
):
    try:
        start_time = time.time()
        last_heartbeat_time = start_time
        logger.info(f"Source Text | {source_text}")

        # === Step 1: DraftAgent ===
        draft_agent = DraftAgent(src_fullname, tgt_fullname, model_platform, model_name, temperature, max_tokens)
        draft_task = asyncio.create_task(draft_agent.run_common(source_text, src_fullname, tgt_fullname))
        # Send heartbeat to keep alive
        async for heartbeat in await_while_task_running_with_heartbeat(draft_task, request_id, model_name):
            yield heartbeat
        draft_translations = await draft_task
        logger.info(f"Elapsed Time: {int(time.time() - start_time)} s. | DraftAgent ｜ {draft_translations}")
        yield f"data: {event(draft_translations, request_id, model_name, 'DraftAgent')}\n\n"
        await asyncio.sleep(0)

        # === Step 2: RefinementAgent ===
        refinement_agent = RefinementAgent(src_fullname, tgt_fullname, model_platform, model_name, temperature, max_tokens)
        candidate_translations = [draft_translations]
        refine_task = asyncio.create_task(refinement_agent.run_common(
            src_fullname, tgt_fullname, source_text, candidate_translations
        ))
        # Send heartbeat to keep alive
        async for heartbeat in await_while_task_running_with_heartbeat(refine_task, request_id, model_name):
            yield heartbeat
        translation, response = await refine_task
        logger.info(f"Elapsed Time: {int(time.time() - start_time)} s. | RefinementAgent ｜ {response}")
        yield f"data: {event(response['analysis'], request_id, model_name, 'RefinementAgent_Analysis')}\n\n"
        await asyncio.sleep(0)
        yield f"data: {event(response['translation'], request_id, model_name, 'Final_Translation', True)}\n\n"
        await asyncio.sleep(0)

        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"[{request_id}] Stream error: {str(e)}")
        error_msg = f"Error: {str(e)}\n"
        yield f"data: {event(error_msg, request_id, model_name, 'Error')}\n\n"
        yield "data: [DONE]\n\n"

if __name__ == '__main__':
    pass
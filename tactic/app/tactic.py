# -*- coding: utf-8 -*-
import os
import json
import time
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

async def run_tactic_light_with_stream(
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
    try:
        start_time = time.time()
        logger.info(f"Source Text | {source_text}")

        # DraftAgent
        draft_agent = DraftAgent(src_fullname, tgt_fullname, model_platform, model_name, temperature, max_tokens)
        draft_translations = await draft_agent.run_common(source_text, src_fullname, tgt_fullname)
        logger.info(f"Elapsed Time: {int(time.time() - start_time)} s. | DraftAgent ｜ {draft_translations}")
        candidate_translations = [draft_translations]

        # Generate JSON data using the event() function and encapsulate it in SSE format
        yield f"data: {event(f'{draft_translations}', request_id, model_name, "DraftAgent")}\n\n"
        await asyncio.sleep(0) # Allow event loop switching

        # RefinementAgent
        refinement_agent = RefinementAgent(src_fullname, tgt_fullname, model_platform, model_name, temperature, max_tokens)
        translation, response = await refinement_agent.run_common(
            src_fullname, tgt_fullname, source_text, candidate_translations
        )
        logger.info(f"Elapsed Time: {int(time.time() - start_time)} s. | RefinementAgent ｜ {response}")
        yield f"data: {event(f'{response['analysis']}', request_id, model_name, "RefinementAgent_Analysis")}\n\n"
        await asyncio.sleep(0)
        yield f"data: {event(f'{response['translation']}', request_id, model_name, "Final_Translation", True)}\n\n"
        await asyncio.sleep(0)

    except Exception as e:
        logger.error(f"[{request_id}] Stream error: {str(e)}")
        yield f"data: {event(f'Error: {str(e)}\n', request_id, model_name, 'Error')}\n\n"

    finally:
        yield "data: [DONE]\n\n"


if __name__ == '__main__':
    pass
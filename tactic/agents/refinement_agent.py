import json
import asyncio
from loguru import logger
from dotenv import load_dotenv, find_dotenv 
from pydantic import BaseModel
from camel.agents import ChatAgent
from tactic.models.model_init import model_init
from tactic.utils import extract_json_with_step, extract_json, extract_html_with_step
from tactic.prompts.multi_agent import REFINEMENT_SYS_PROMPT, REFINEMENT_PROMPT, REFINEMENT_PROMPT_COMMON

load_dotenv(find_dotenv())


class RefinementAgent:
    def __init__(
        self, 
        source_language: str, 
        target_language: str, 
        model_platform: str, 
        model_name: str,
        temperature: float=0.6, 
        max_tokens: int=2048,
        port: int=8000, 
    ):
        # Define the JSON Schema using Pydantic
        class TranslationTemplate(BaseModel):
            analysis: str
            translation: str
        # Generate the JSON Schema
        json_schema = TranslationTemplate.model_json_schema()
        self.response_format = TranslationTemplate
        
        self.model_config = {
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if model_platform.upper() == 'VLLM':
            self.model_config['extra_body'] = {"guided_json": json_schema}
        elif model_platform.upper() == 'DEEPSEEK':
            self.model_config['response_format'] = {'type': 'json_object'}
            
        self.model = model_init(model_platform, model_name, port, self.model_config)  
        self.sys_prompt = REFINEMENT_SYS_PROMPT.format(
            source_language=source_language, 
            target_language=target_language, 
        )
        self.prompt = REFINEMENT_PROMPT
        self.agent=ChatAgent(
            system_message=self.sys_prompt,
            model=self.model,
            single_iteration=True,
            token_limit=8192,
        )

    async def run_common(
        self, 
        source_language, 
        target_language, 
        source_text, 
        candidate_translations, 
        pre_translation_result="", 
        context_analysis="",
    ):
        self.prompt = REFINEMENT_PROMPT_COMMON
        formatted_user_prompt = self.prompt.format(
            source_language=source_language, 
            source_text=source_text, 
            target_language=target_language, 
            candidate_translations=candidate_translations, 
            pre_translation_result=pre_translation_result, 
            context_analysis=context_analysis, 
        )
        
        response = extract_json_with_step(self.agent, formatted_user_prompt, self.response_format, tag=["analysis", "translation"])
        self.agent.init_messages()
        
        return response['translation'], response

    @staticmethod
    def agent_name():
        return "RefinementAgent"

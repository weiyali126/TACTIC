# -*- coding: utf-8 -*-
import os
import re

import openai
from loguru import logger
from openai import BadRequestError

from tower_eval.models.exceptions import GenerationException
from tower_eval.models.inference_handler import Generator
from tower_eval.utils import generate_with_retries

from tower_eval.config import OPENAI_API_KEY, OPENAI_API_BASE_URL

class OpenAI(Generator):
    """OpenAI GPT completion Wrapper.

    Args:
        api_org (str): The Org ID for OpenAI org
        api_key (str): OpenAI API Key
        api_base (str, optional): OpenAI API Base URL. Defaults to "https://api.openai.com/v1".
        api_version (str, optional): OpenAI API Version. Defaults to None.
        api_type (str, optional): OpenAI API Type. Defaults to OpenAI.
        retry_max_attempts (int, optional): Maximum number of retries. Defaults to 1.
        retry_max_interval (int, optional): Maximum interval between retries. Defaults to 10.
        retry_min_interval (int, optional): Minimum interval between retries. Defaults to 4.
        retry_multiplier (int, optional): Multiplier for the retry interval. Defaults to 1.
    """

    def __init__(
        self,
        api_key: str = OPENAI_API_KEY,
        api_base: str = OPENAI_API_BASE_URL,
        model: str = "",
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        retry_max_attempts: int = 1,
        retry_max_interval: int = 10,
        retry_min_interval: int = 4,
        retry_multiplier: int = 1,
        stop_sequences: list[str] = [],
        run_async: bool = False,
        system_prompt: str = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.run_async = False  # only sync calls are supported
        # Set openai settings
        model = kwargs.get("model", model)
        temperature = kwargs.get("temperature", temperature)
        top_p = kwargs.get("top_p", top_p)
        max_tokens = kwargs.get("max_tokens", max_tokens)
        frequency_penalty = kwargs.get("frequency_penalty", frequency_penalty)
        presence_penalty = kwargs.get("presence_penalty", presence_penalty)
        stop_sequences = kwargs.get("stop_sequences", stop_sequences)
        self.system_prompt = system_prompt
        self.run_async = run_async
        self.openai_args = {
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "max_completion_tokens": max_tokens,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stop": stop_sequences,
        }

        self.retry_max_attempts = kwargs.get("retry_max_attempts", retry_max_attempts)
        self.retry_max_interval = kwargs.get("retry_max_interval", retry_max_interval)
        self.retry_min_interval = kwargs.get("retry_min_interval", retry_min_interval)
        self.retry_multiplier = kwargs.get("retry_multiplier", retry_multiplier)
        self.model_max_tokens = {
            "gpt-3.5-turbo": 4096,
            "gpt-4": 8192,
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
        }.get(model, 32000)
        self.client = openai.Client(api_key=api_key, base_url=api_base)

    def _generate(self, input_line: str) -> str:
        """It calls the Chat completion function of OpenAI.

        Args:
            prompt (str): Prompt for the OpenAI model


        Returns:
            str: Returns the response used.
        """
        try:
            if self.system_prompt is not None:
                prompt = {
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": input_line},
                    ]
                }
            else:
                prompt = {"messages": [{"role": "user", "content": input_line}]}
            response = generate_with_retries(
                retry_function=self.client.chat.completions.create,
                model_args=self.openai_args | prompt,
                retry_max_attempts=self.retry_max_attempts,
                retry_multiplier=self.retry_multiplier,
                retry_min_interval=self.retry_min_interval,
                retry_max_interval=self.retry_max_interval,
            )
        except Exception as e:
            if type(e) == BadRequestError:
                response = self._handle_excessive_tokens_error(e, prompt)
            else:
                raise GenerationException(str(e))

        response = response.choices[0].message.content
        return response

    # def _handle_excessive_tokens_error(self, e: InvalidRequestError, prompt: dict):
    def _handle_excessive_tokens_error(self, e: BadRequestError, prompt: dict):
        logger.error(
            f'Handling Open AI excessive tokens requested error by decreasing max tokens for this request. ("{str(e)}")'
        )
        requested_tokens = int(re.findall(r"you requested (\d+) tokens", str(e))[0])
        excessive_tokens = requested_tokens - self.model_max_tokens
        old_max_tokens = self.openai_args["max_completion_tokens"]
        new_max_tokens = old_max_tokens - excessive_tokens
        self.openai_args["max_completion_tokens"] = new_max_tokens
        logger.warning(
            f"Decreased max tokens from {old_max_tokens} to {new_max_tokens}."
        )
        response = generate_with_retries(
            retry_function=self.client.chat.completions.create,
            model_args=self.openai_args | prompt,
            retry_max_attempts=self.retry_max_attempts,
            retry_multiplier=self.retry_multiplier,
            retry_min_interval=self.retry_min_interval,
            retry_max_interval=self.retry_max_interval,
        )
        logger.warning(f"Restoring max tokens to {old_max_tokens}.")
        self.openai_args["max_completion_tokens"] = old_max_tokens

        return response

    @staticmethod
    def model_name():
        return "open-ai"

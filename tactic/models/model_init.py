import time
import requests
from loguru import logger
from pydantic import BaseModel
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from dotenv import load_dotenv, find_dotenv 
from tactic.config import API_KEY_DICT, BASE_URL_DICT

load_dotenv(find_dotenv())


def model_init(model_platform, model_name, port, model_config):
    """
    Initialize model class.
    model_platform and model_type: camel.types.enums
    """
    
    if model_platform.upper() == 'VLLM':
        model = ModelFactory.create(
            model_platform = ModelPlatformType.VLLM,
            model_type = model_name,
            url = f"http://0.0.0.0:{port}/v1",
            model_config_dict = model_config,
        )
    else:
        # Map model_type
        model_type_dict = {m.value: m.name for m in ModelType}
        model_type = model_type_dict[model_name.lower()]
        model = ModelFactory.create(
            model_platform = ModelPlatformType[model_platform.upper()],
            model_type = ModelType[model_type],
            api_key = API_KEY_DICT[model_platform.upper()],
            url = BASE_URL_DICT[model_platform.upper()],
            model_config_dict = model_config,
        )
    return model




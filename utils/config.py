import os, json

from os import getenv
from typing import Optional, Literal
from dotenv import load_dotenv
from dataclasses import dataclass
from pydantic import BaseModel
from vllm import AsyncEngineArgs

from utils.logger import get_logger

load_dotenv()
CONFIG_PATH = getenv("CONFIG_PATH")

logger = get_logger(__name__)


# 原始 config 文件的数据结构

class DevicesConfig(BaseModel):
    gpu_ids: list = [0]
    weight: float = 1.0


class ModelConfig(BaseModel):
    devices: DevicesConfig = DevicesConfig()
    engine_args: dict


class ConfigSchema(BaseModel):
    common: Optional[dict] = None
    models: dict[str, list[ModelConfig]]


# 解析后的数据结构

@dataclass
class EngineConfig:
    model_name: str
    devices: DevicesConfig
    engine_args: AsyncEngineArgs


class Config:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if Config._initialized:
            return

        if not os.path.exists(CONFIG_PATH):
            raise FileNotFoundError(f"Config file {CONFIG_PATH} not found")

        with open(CONFIG_PATH) as f:
            # 读取配置
            config: ConfigSchema = json.load(f)
            self.common: dict = config["common"] if "common" in config else {}
            self.models: dict[str, list[dict]] = config["models"]
            self.engine_configs: list[EngineConfig] = []

            # 创建配置
            for model_name, model_configs in self.models.items():
                model_config_list: list[ModelConfig] = [ModelConfig(**model_config) for model_config in model_configs]
                if len(model_config_list) == 0:
                    raise ValueError(f"Model {model_name} has no config"
                                     f"Please check your config file and make sure it has at least one config")

                # 遍历加载该模型的分布式部署配置
                total_weight = sum(model_config.devices.weight for model_config in model_config_list)
                assert total_weight > 0, f"The sum of the weights of the model configurations of {model_name} must be greater than 0"
                for model_config in model_config_list:
                    assert model_config.devices.weight > 0, f"The weight of the model configuration of {model_name} must be greater than 0"
                    model_config.devices.weight /= total_weight
                    engine_config = EngineConfig(
                            model_name=model_name,
                            devices=model_config.devices,
                            engine_args=AsyncEngineArgs(**model_config.engine_args)
                    )

                    # 补充配置
                    engine_config.engine_args.disable_log_requests = True

                    self.engine_configs.append(engine_config)

        Config._initialized = True
        logger.info(f"Config loaded successfully, total {len(self.engine_configs)} engines.")

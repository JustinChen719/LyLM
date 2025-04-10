import os, json

from os import getenv
from dotenv import load_dotenv
from dataclasses import dataclass
from vllm import AsyncEngineArgs

load_dotenv()
CONFIG_PATH = getenv("CONFIG_PATH")


@dataclass
class EngineConfig:
    model_name: str
    devices: str
    engine_args: AsyncEngineArgs


class Config:
    _instance = None
    _initialized = False

    def __new__(cls,
                *args,
                **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        '''
        Config类的单例
        '''
        if Config._initialized:
            return

        if not os.path.exists(CONFIG_PATH):
            raise FileNotFoundError(f"Config file {CONFIG_PATH} not found")
        self.engine_configs: list[EngineConfig] = []
        with open(CONFIG_PATH) as f:
            # 读取配置
            config = json.load(f)
            required = ["global", "models"]
            for attr in required:
                if attr not in config:
                    raise AttributeError(f"Config file is missing '{attr}' attribute")

            self._global: dict = config["global"]
            self.models: dict[str, dict] = config["models"]

            # 创建配置
            for model_name, model_config in self.models.items():
                gpu_ids = "0"
                if "gpu_ids" in model_config:
                    gpu_ids = model_config["gpu_ids"]
                    model_config.pop("gpu_ids")
                engin_config = EngineConfig(
                    model_name=model_name,
                    gpu_ids=gpu_ids,
                    engine_args=AsyncEngineArgs(**model_config)
                )
                self.engine_configs.append(engin_config)

        Config._initialized = True
        print(f"Config loaded successfully")

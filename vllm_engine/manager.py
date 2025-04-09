import uuid
from typing import AsyncGenerator

from core.configs import Config
from core.schemas import *
from vllm_engine.engine import Engine


class Manager:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Manager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if Manager._initialized:
            return

        self.engines: dict[str, list[Engine]] = {}
        self._init_engines()
        Manager._initialized = True

    def _init_engines(self):
        config = Config()
        print(f"Initializing {len(config.engine_configs)} engines ...")
        for engine_config in config.engine_configs:
            model_name = engine_config.model_name
            if model_name not in self.engines:
                self.engines[model_name] = []
            self.engines[model_name].append(Engine(engine_config))

    async def generate(self, model_name: str, history: list, max_tokens, temperature, top_p: float, stream: bool) \
            -> AsyncGenerator[str, None]:

        if model_name not in self.engines:
            raise KeyError(f"Model {model_name} not found")

        # todo 负载均衡、调度机制， 当前默认取第一个
        engine = self.engines[model_name][0]
        request_id = str(uuid.uuid4())

        # todo 多个输出序列
        try:
            
            async for outputs in engine.generate(request_id, history, max_tokens, temperature, top_p):
                output = outputs[0]
                response = CompletionResponse(
                    model=model_name,
                    choices=[CompletionResponseStreamChoice(
                        index=output.index,
                        delta=StreamMessage(content=output.text),
                        finish_reason=output.finish_reason
                    )],
                )
                yield response.model_dump_json(exclude_unset=True)

        except Exception as e:
            print(e)
            response = CompletionResponse(
                model=model_name,
                choices=[CompletionResponseStreamChoice(index=0, delta=StreamMessage(content=""), finish_reason="abort")],
            )
            yield response.model_dump_json(exclude_unset=True)
        finally:
            yield "[DONE]"


def get_manager() -> Manager:
    return Manager()

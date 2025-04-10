import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator, Any

from vllm import CompletionOutput

from core.scheduler import Scheduler
from utils import get_logger
from utils.config import Config
from utils.types.chat_completion_chunk import *
from utils.types.chat_completion import *
from utils.types.schemas import *

logger = get_logger(__name__)


class Manager:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Manager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        '''
        管理器，作为 FastApi 和 vLLM 引擎的中介。

        使用调度器Scheduler，根据用户提供的权重配置来控制转发

        ps: 由于当前只支持单机多卡的模式，没有涉及多节点分布式调度，可能一般更多地会使用网关层比如 Nginx 来实现。
        '''
        if Manager._initialized:
            return

        self.scheduler = Scheduler()
        self.executor = ThreadPoolExecutor(max_workers=8)
        self._init_engines()
        Manager._initialized = True

    def _init_engines(self):
        config = Config()
        for engine_config in config.engine_configs:
            self.scheduler.add_engine(engine_config)

    def get_model_list(self) -> ModelList:
        '''
        获取模型列表
        '''
        model_name_list = self.scheduler.engines.keys()
        model_list = []
        for model_name in model_name_list:
            model_list.append(ModelInfo(id=model_name))

        # debug
        self.scheduler.print_status()

        return ModelList(data=model_list)

    async def generate(
            self,
            model_name: str,
            history: list,
            max_tokens,
            temperature,
            top_p: float,
            n: int,
    ) -> AsyncGenerator[ChatCompletion, Any]:
        # 生成唯一请求ID并注册到调度器
        request_id = str(uuid.uuid4())
        engine = await self.scheduler.add_request(request_id, model_name)
        if not engine:
            raise KeyError("Model not found")

        failed = False
        try:
            # 对 final_outputs 进行更新
            final_outputs: list[Optional[CompletionOutput]] = [None] * n
            async for outputs in engine.generate(request_id, history, max_tokens, temperature, top_p, n):
                for output in outputs:
                    if output.finish_reason is not None:
                        final_outputs[output.index] = output

            # 封装、返回
            choices: list[Optional[ChatCompletionChoice]] = []
            for output in final_outputs:
                if output is None:
                    continue

                choices.append(ChatCompletionChoice(
                        index=output.index,
                        message=ChatCompletionMessage(content=output.text),
                        finish_reason=output.finish_reason
                ))
            yield ChatCompletion(id=request_id, model=model_name, choices=choices, created=int(time.time()))

        except Exception as e:
            logger.error(f"error: {e}")
            failed = True
            yield ChatCompletion(id=request_id, model=model_name, choices=[], created=int(time.time()))

        finally:
            await self.scheduler.remove_request(request_id, failed)

    async def generate_async(
            self,
            model_name: str,
            history: list,
            max_tokens,
            temperature,
            top_p: float,
            n: int,
    ) -> AsyncGenerator[str, None]:
        # 生成唯一请求ID并注册到调度器
        request_id = str(uuid.uuid4())
        engine = await self.scheduler.add_request(request_id, model_name)
        if not engine:
            raise KeyError("Model not found")

        failed = False
        try:
            num_new_tokens = [0] * n
            async for outputs in engine.generate(request_id, history, max_tokens, temperature, top_p, n):
                response = ChatCompletionChunk(
                        id=request_id,
                        model=model_name,
                        choices=[],
                        created=int(time.time()),
                )
                for output in outputs:
                    response.choices.append(ChatCompletionChunkChoice(
                            index=output.index,
                            delta=ChatCompletionChunkDelta(content=output.text[num_new_tokens[output.index]:]),
                            finish_reason=output.finish_reason
                    ))
                    num_new_tokens[output.index] = len(output.text)
                yield response.model_dump_json(exclude_unset=True)

        except Exception as e:
            logger.error(f"error: {e}")
            failed = True
        finally:
            yield "[DONE]"
            await self.scheduler.remove_request(request_id, failed)


def get_manager() -> Manager:
    return Manager()

import os

from typing import AsyncGenerator
from transformers import AutoTokenizer
from vllm import AsyncLLMEngine, SamplingParams, CompletionOutput

from utils.config import EngineConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class Engine:
    def __init__(self, engine_config: EngineConfig):
        logger.info(f"\n=== > Init LLM Engine: {engine_config.model_name} < ==="
                    f"\n\tModel path: {engine_config.engine_args.model}"
                    f"\n\tDevices: {engine_config.devices.gpu_ids}"
                    f"\n\tWeight: {engine_config.devices.weight}")

        self.engine_config = engine_config
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, engine_config.devices.gpu_ids))
        self.tokenizer = AutoTokenizer.from_pretrained(engine_config.engine_args.model, local_files_only=True)
        self.engine = AsyncLLMEngine.from_engine_args(engine_args=self.engine_config.engine_args)

    async def generate(
            self,
            request_id: str,
            history: list,
            max_tokens: int,
            temperature: float,
            top_p: float,
            n: int
    ) -> AsyncGenerator[list[CompletionOutput], None]:

        try:
            sampling_params = SamplingParams(
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=n,
                    # best_of=n, # 截至 vLLM 0.8.3，v1 不支持 best_of 参数，并且 n 的值设置超过 2 时，只返回 2 个
                    stop_token_ids=[self.tokenizer.eos_token_id],
            )
            prompt = self.tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)

            async for output in self.engine.generate(prompt=prompt, sampling_params=sampling_params, request_id=request_id):
                yield output.outputs
        finally:
            await self.engine.abort(request_id)

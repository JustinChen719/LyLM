import os, logging
from typing import AsyncGenerator
from transformers import AutoTokenizer
from vllm import AsyncLLMEngine, SamplingParams, CompletionOutput

from utils.config import EngineConfig

logger = logging.getLogger(__name__)


class Engine:
    def __init__(self, engine_config: EngineConfig):
        logger.info(f"Init LLM Engine: {engine_config.engine_args.model}")
        self.engine_config = engine_config
        os.environ["CUDA_VISIBLE_DEVICES"] = engine_config.gpu_ids
        self.tokenizer = AutoTokenizer.from_pretrained(engine_config.engine_args.model, local_files_only=True)
        self.engine = AsyncLLMEngine.from_engine_args(engine_args=self.engine_config.engine_args)

    async def generate(
            self,
            request_id: str,
            history: list,
            max_tokens: int,
            temperature: float,
            top_p: float
    ) -> AsyncGenerator[list[CompletionOutput], None]:

        try:
            sampling_params = SamplingParams(
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=1,
                    stop_token_ids=[self.tokenizer.eos_token_id],
            )
            prompt = self.tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)

            async for output in self.engine.generate(prompt=prompt, sampling_params=sampling_params, request_id=request_id):
                yield output.outputs
                # delta = output.outputs[0].text[response_len:]
                # finish_reason = output.outputs[0].finish_reason
                # response_len = len(output.outputs[0].text)
                # yield delta
        finally:
            await self.engine.abort(request_id)

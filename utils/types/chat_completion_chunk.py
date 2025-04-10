from pydantic import BaseModel
from typing import Literal, Optional, Union


class ChatCompletionChunkDelta(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[Literal["stop", "length", "abort"]]


class ChatCompletionChunk(BaseModel):
    id: str
    model: str
    choices: list[Union[ChatCompletionChunkChoice, None]]
    created: int
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"

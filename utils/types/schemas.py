import time

from pydantic import BaseModel, Field
from typing import Literal, Optional, Union

'''
{
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1677652288,
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "  Hello there, how may I assist you today?",
        },
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 9,
        "completion_tokens": 12,
        "total_tokens": 21
    }
}
'''

'''
completion:

{
    "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
    "object": "text_completion",
    "created": 1589478378,
    "model": "text-davinci-003",
    "choices": [{
            "index": 0,
            "text": "  This is indeed a test",
            "logprobs": null,
            "finish_reason": "length"
    }],
    "usage": {
        "prompt_tokens": 5,
        "completion_tokens": 7,
        "total_tokens": 12
    }
}
'''


### 模型信息 ###

class ModelInfo(BaseModel):
    id: str
    owned_by: str = "owner"
    permission: Optional[list] = []
    object: str = "model"


class ModelList(BaseModel):
    data: list[ModelInfo] = []
    object: str = "list"


### 基础信息 ###

class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class CompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Literal["stop", "length", "abort"]


class StreamMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None


class CompletionResponseStreamChoice(BaseModel):
    index: int
    delta: StreamMessage
    finish_reason: Optional[Literal["stop", "length", "abort"]]


### 请求 ###

class CompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.75
    top_p: Optional[float] = 0.95
    stream: Optional[bool] = False


### 响应 ###

class CompletionResponse(BaseModel):
    model: str
    choices: list[Union[CompletionResponseChoice, CompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    object: str = "text_completion"

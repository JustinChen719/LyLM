from pydantic import BaseModel
from typing import Literal, Optional, Union


class ChatCompletionMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter", "function_call"]]


class ChatCompletion(BaseModel):
    id: str
    model: str
    choices: list[Union[ChatCompletionChoice, None]]
    created: int
    object: Literal["chat.completion"] = "chat.completion"

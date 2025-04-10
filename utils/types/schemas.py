from pydantic import BaseModel, Field
from typing import Literal, Optional, Union


### 模型信息 ###

class ModelInfo(BaseModel):
    id: str
    owned_by: str = "owner"
    permission: Optional[list] = []
    object: str = "model"


class ModelList(BaseModel):
    data: list[ModelInfo] = []
    object: str = "list"


### 请求 ###

class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: Optional[str] = None


class CompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.75
    top_p: Optional[float] = 0.95
    n: Optional[int] = 1
    stream: Optional[bool] = False

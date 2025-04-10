from fastapi import FastAPI, Depends, HTTPException
from starlette.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from core.manager import Manager, get_manager
from utils.types.chat_completion_chunk import ChatCompletionChunk
from utils.types.chat_completion import ChatCompletion
from utils.types.schemas import *

app = FastAPI()
app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
)


@app.get("/v1/models", response_model=ModelList)
async def list_models(manager: Manager = Depends(get_manager)):
    return manager.get_model_list()


@app.post("/v1/chat/completions", response_model=Union[ChatCompletionChunk, ChatCompletion])
async def generate(
        request: CompletionRequest,
        manager: Manager = Depends(get_manager)
):
    try:
        model, messages = request.model, request.messages
        max_tokens, temperature, top_p = request.max_tokens, request.temperature, request.top_p
        n, stream = request.n, request.stream

        if not stream:
            return await anext(manager.generate(model, messages, max_tokens, temperature, top_p, n))

        response = manager.generate_async(model, messages, max_tokens, temperature, top_p, n)
        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        return EventSourceResponse(response, headers=headers)
    except KeyError:
        raise HTTPException(status_code=404, detail="Model not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=18000)

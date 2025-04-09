from fastapi import FastAPI, Body, Depends, HTTPException
from starlette.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from core.schemas import *
from vllm_engine.manager import Manager, get_manager

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
    model_list = []
    for model_name in manager.engines.keys():
        model_list.append(ModelInfo(id=model_name))
    return ModelList(data=model_list)


@app.post("/v1/chat/completions", response_model=CompletionResponse)
async def generate(request: CompletionRequest,
                   manager: Manager = Depends(get_manager)):
    try:
        model, messages = request.model, request.messages
        max_tokens, temperature, top_p = request.max_tokens, request.temperature, request.top_p
        stream = request.stream

        response = manager.generate(model, messages, max_tokens, temperature, top_p, stream)
        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        return EventSourceResponse(response, headers=headers)
    except KeyError:
        raise HTTPException(status_code=404, detail="Model not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=18000)

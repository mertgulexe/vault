from langchain_core.runnables import Runnable
from langchain_core.callbacks import BaseCallbackHandler
from fastapi import FastAPI, Request, Depends
from sse_starlette.sse import EventSourceResponse
from langserve.serialization import WellKnownLCSerializer
from typing import List
from sqlalchemy.orm import Session

import crud
import models
import schemas
from chains import (
    simple_chain,
    formatted_chain,
    history_chain,
    rag_chain,
    filtered_rag_chain
)
from database import SessionLocal, engine
from callbacks import LogResponseCallback
from prompts import format_chat_history


models.Base.metadata.create_all(bind=engine)

app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def generate_stream(
    input_data: schemas.BaseModel,
    runnable: Runnable,
    callbacks: List[BaseCallbackHandler] = []
):
    for output in runnable.stream(
        input=input_data.dict(),
        config={
            "callbacks": callbacks
        }
    ): 
        data = WellKnownLCSerializer().dumps(output).decode("utf-8")
        yield {
            "data": data,
            "event": "data"
        } 
    yield {
        "event": "end"
    }


@app.post("/simple/stream")
async def simple_stream(request: Request):
    data = await request.json()
    user_request = schemas.UserRequest(**data["input"])
    return EventSourceResponse(generate_stream(user_request, simple_chain))


@app.post("/formatted/stream")
async def formatted_stream(request: Request):
    # TODO: use the formatted_chain to implement the "/formatted/stream" endpoint.
    data = await request.json()
    user_request = schemas.UserRequest(**data["input"])
    return EventSourceResponse(
        generate_stream(
            input_data=user_request,
            runnable=formatted_chain,
        )
    )   


@app.post("/history/stream")
async def history_stream(request: Request, db: Session = Depends(get_db)):  
    # TODO: Let's implement the "/history/stream" endpoint. The endpoint should follow those steps:
    # - The endpoint receives the request
    # - The request is parsed into a user request
    # - The user request is used to pull the chat history of the user
    # - We add as part of the user history the current question by using add_message.
    # - We create an instance of HistoryInput by using format_chat_history.
    # - We use the history input within the history chain.
    data = await request.json()
    user_request = schemas.UserRequest(**data["input"])
    user_name = user_request.username
    chat_history = crud.get_user_chat_history(
        db=db,
        username=user_name
    )
    user_question = schemas.MessageBase(
        message=user_request.question,
        role_type="Human",
        user=user_name,
    )
    crud.add_message(db=db, message=user_question, username=user_name)
    chat_history_formatted = schemas.HistoryInput(
        chat_history=format_chat_history(chat_history),
        question=user_request.question
    )
    return EventSourceResponse(
        generate_stream(
            input_data=chat_history_formatted,
            runnable=history_chain,
            callbacks=[LogResponseCallback(user_request=user_request, db=db)]
        )
    )


@app.post("/rag/stream")
async def rag_stream(request: Request, db: Session = Depends(get_db)):  
    # TODO: Let's implement the "/rag/stream" endpoint. The endpoint should follow those steps:
    # - The endpoint receives the request
    # - The request is parsed into a user request
    # - The user request is used to pull the chat history of the user
    # - We add as part of the user history the current question by using add_message.
    # - We create an instance of HistoryInput by using format_chat_history.
    # - We use the history input within the rag chain.
    data = await request.json()
    user_request = schemas.UserRequest(**data["input"])
    user_name = user_request.username
    chat_history = crud.get_user_chat_history(
        db=db,
        username=user_name
    )
    user_question = schemas.MessageBase(
        message=user_request.question,
        role_type="Human",
        user=user_name,
    )
    crud.add_message(db=db, message=user_question, username=user_name)
    chat_history_formatted = schemas.HistoryInput(
        chat_history=format_chat_history(chat_history),
        question=user_question.message
    )
    return EventSourceResponse(
        generate_stream(
            input_data=chat_history_formatted,
            runnable=rag_chain,
            callbacks=[LogResponseCallback(user_request=user_request, db=db)]
        )
    )


@app.post("/filtered_rag/stream")
async def filtered_rag_stream(request: Request, db: Session = Depends(get_db)):  
    # TODO: Let's implement the "/filtered_rag/stream" endpoint. The endpoint should follow those steps:
    # - The endpoint receives the request
    # - The request is parsed into a user request
    # - The user request is used to pull the chat history of the user
    # - We add as part of the user history the current question by using add_message.
    # - We create an instance of HistoryInput by using format_chat_history.
    # - We use the history input within the filtered rag chain.
    data = await request.json()
    user_request = schemas.UserRequest(**data["input"])
    user_name = user_request.username
    chat_history = crud.get_user_chat_history(
        db=db,
        username=user_name
    )
    user_question = schemas.MessageBase(
        message=user_request.question,
        role_type="Human",
        user=user_name,
    )
    crud.add_message(db=db, message=user_question, username=user_name)
    chat_history_formatted = schemas.HistoryInput(
        chat_history=format_chat_history(chat_history),
        question=user_question.message
    )
    return EventSourceResponse(
        generate_stream(
            input_data=chat_history_formatted,
            runnable=filtered_rag_chain,
            callbacks=[LogResponseCallback(user_request=user_request, db=db)]
        )
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="localhost", reload=True,  port=8001)

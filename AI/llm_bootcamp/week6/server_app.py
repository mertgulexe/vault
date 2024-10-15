from fastapi import FastAPI
from pydantic import BaseModel
from langserve import add_routes
from utils import DataIndexer, QARetriever, INDEX_NAME
# from retrieval.conversation_qa import QARetriever
# from data_processing.data_indexing import DataIndexer

data_indexer = DataIndexer(index_name=INDEX_NAME)
retriever = data_indexer.get_retriever()
qa_retriever = QARetriever(retriever=retriever).get_chain()


class UserInput(BaseModel):
    input: str


qa_retriever = qa_retriever.with_types(input_type=UserInput)
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server to answer question"
)
add_routes(
    app=app,
    runnable=qa_retriever,
    path="/rag"

)
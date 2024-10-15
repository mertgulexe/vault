## 1. Implementing the Indexing Pipeline
# dataloading.py
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROMPT_LIST = """Generate a numbered list of 3 hypothetical questions that \
the following document could be used to answer.\nDocument: {doc}"""


class DataLoader:
    def __init__(
        self,
        file_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.pdf_loader = PyPDFLoader(file_path=file_path)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    
    def get_docs(self):
        return self.pdf_loader.load_and_split(text_splitter=self.text_splitter)


# data_mapping.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import NumberedListOutputParser


class DataMapper:
    def __init__(self) -> None:
        model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)
        prompt = PromptTemplate.from_template(template=PROMPT_LIST)
        parser = NumberedListOutputParser()
        self.chain = prompt | model | parser
    
    def transform(self, docs):
        transformed_docs = []
        all_questions = self.chain.batch(
            inputs=[
                {"doc": doc.page_content} for doc in docs
            ]
        )
        for question_list, doc in zip(all_questions, docs):
            for question in question_list:
                transformed_docs.append((question, doc))
        return transformed_docs


# data_indexing.py
import os
import uuid
from dotenv import load_dotenv
from pinecone import ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv()


class DataIndexer:
    def __init__(self, index_name: str) -> None:
        self.embeddings = OpenAIEmbeddings(
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        self.pinecone_client = Pinecone()
        self.index = None
        if index_name not in self.pinecone_client.list_indexes().names():
            self.pinecone_client.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        self.index = self.pinecone_client.Index(index_name)
        # this variable will be used for RAG later:
        self.vectorstore = PineconeVectorStore(
            index=self.pinecone_client.Index(index_name),
            embedding=self.embeddings
        )
    
    def index_data(self, transformed_docs, batch_size: int = 32):
        for i in range(0, len(transformed_docs), batch_size):
            batch = transformed_docs[i: i + batch_size]
            vector_values = self.embeddings.embed_documents(
                texts=[question for question, doc in batch]
            )
            vector_ids = [str(uuid.uuid4()) for i in range(len(batch))]
            metadata_list = [
                {
                    "text": doc.page_content,
                    **doc.metadata
                } for question, doc in batch
            ]
            for metadata, (question, doc) in zip(metadata_list, batch):
                metadata["questions"] = question
            vectors = [
                {
                    "id": vi,  # vector id
                    "values": vv,  # vector value
                    "metadata": md  # metadata
                } for vi, vv, md in zip(vector_ids, vector_values, metadata_list)
            ]
            try:
                upsert_response = self.index.upsert(vectors=vectors)
                print("\tINDEXING SUCCESSFUL -", upsert_response)
            except Exception as error:
                print("\tINDEXING FAILED -", error)
    
    def get_retriever(self):
        return self.vectorstore.as_retriever()


# indexing_pipeline.py
# from data_processing.data_loading import DataLoader
# from data_processing.data_mapping import DataMapper
# from data_processing.data_indexing import DataIndexer


FILE_PATH = "/home/mert/Documents/Mert Gul - CV.pdf"
INDEX_NAME = "mert-resume"

def index_my_data(
        file_path: str = FILE_PATH,
        index_name: str = INDEX_NAME,
        chunk_size: int = 400,
        chunk_verlap: int = 80) -> None:
    data_loader = DataLoader(
        file_path=file_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_verlap
    )
    print("1. Data loader is loaded.")
    data_mapper = DataMapper()
    print("2. Data mapper is loaded.")
    data_indexer = DataIndexer(index_name=index_name)
    print("3. Data indexer is loaded.")
    docs_list = data_loader.get_docs()
    print("4. Documents has been retrieved.")
    transformed_doc_list = data_mapper.transform(docs=docs_list)
    print("5. Documents has been transformed.")
    data_indexer.index_data(transformed_docs=transformed_doc_list)
    print("6. Data has been indexed successfully.")


# 2.Implementing the Retrieval API
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


SYSTEM_PROMPT = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, say that you don't know. \
Use three sentences maximum and keep the answer concise.{context}"""


class QARetriever:
    def __init__(self, retriever) -> None:
        self.retriever = retriever
        llm = ChatOpenAI(model_name="gpt-4o-mini")
        qa_prompt = ChatPromptTemplate.from_messages(
            messages=[
                ("system", SYSTEM_PROMPT),
                ("human", "{input}")
            ]
        )
        qa_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=qa_prompt
        )
        self.retrieval_chain = create_retrieval_chain(
            retriever=self.retriever,
            combine_docs_chain=qa_chain            
        )
    
    def get_chain(self):
        return self.retrieval_chain

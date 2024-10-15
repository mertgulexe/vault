import os
import uuid
from dotenv import load_dotenv
from pathlib import Path

from pinecone import ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings


current_dir = Path(__file__).resolve().parent
load_dotenv(dotenv_path=os.path.join(str(current_dir).split("Backend")[0], ".env"))


class DataIndexer:
    def __init__(self, index_name="mygithub-repo") -> None:
        self.source_file = os.path.join(current_dir, "sources.txt")
        # TODO: choose your embedding model
        # Mert: The first one is from OpenAI API:
        self.embedding_client = OpenAIEmbeddings(
            api_key=os.environ.get('OPENAI_API_KEY')
        )
        # self.embedding_client = HuggingFaceEmbeddings(
        #     model_name="sentence-transformers/all-MiniLM-L6-v2",
        #     model_kwargs={"token": os.environ["HF_TOKEN"]},
        # )
        self.index_name = index_name
        self.pinecone_client = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

        if index_name not in self.pinecone_client.list_indexes().names():
            # TODO: create your index if it doesn't exist. Use the create_index function.
            # Make sure to choose the dimension that corresponds to your embedding model
            self.pinecone_client.create_index(
                name=self.index_name,
                dimension=1536,  # 384,  # Mert: For "sentence-transformers/all-MiniLM-L6-v2",
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        self.index = self.pinecone_client.Index(self.index_name)
        # TODO: make sure to build the index.
        self.source_index = self.get_source_index()

    def get_source_index(self):
        if not os.path.isfile(self.source_file):
            print("No source file.")
            return None

        print("Create source index.")

        with open(self.source_file, "r") as file:
            sources = file.readlines()

        sources = [s.rstrip("\n") for s in sources]
        vectorstore = Chroma.from_texts(texts=sources, embedding=self.embedding_client)
        return vectorstore

    def index_data(self, docs, batch_size=32):

        with open(self.source_file, "a") as file:
            for doc in docs:
                file.writelines(doc.metadata["source"] + "\n")

        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]

            # TODO: create a list of the vector representations of each text data in the batch
            # TODO: choose your embedding model
            values = self.embedding_client.embed_documents([
                doc.page_content for doc in batch
            ])

            # TODO: create a list of unique identifiers for each element in the batch with the uuid package.
            vector_ids = [str(uuid.uuid4()) for _ in range(len(batch))]

            # TODO: create a list of dictionaries representing the metadata. Capture the text data
            # with the "text" key, and make sure to capture the rest of the doc.metadata.
            metadatas = [
                {
                    "text": doc.page_content,
                    **doc.metadata
                } for doc in batch
            ]

            # create a list of dictionaries with keys "id" (the unique identifiers), "values"
            # (the vector representation), and "metadata" (the metadata).
            vectors = [
                {
                    "id": vector_id,
                    "values": value,
                    "metadata": metadata
                } for vector_id, value, metadata in zip(
                    vector_ids,
                    values,
                    metadatas
                )
            ]

            try:
                # TODO: Use the function upsert to upload the data to the database.
                upsert_response = self.index.upsert(vectors=vectors)
                print("OK - Successful upsert response:", upsert_response)
            except Exception as e:
                print("ERROR - Failed upsert response:", e)

    def search(self, text_query, top_k=3, hybrid_search=False):

        search_filter = None
        if hybrid_search and self.source_index:
            # I implemented the filtering process to pull the 50 most relevant file names
            # to the question. Make sure to adjust this number as you see fit.
            source_docs = self.source_index.similarity_search(query=text_query, k=50)
            search_filter = {
                "source": {"$in": [doc.page_content for doc in source_docs]}
            }

        # TODO: embed the text_query by using the embedding model
        # TODO: choose your embedding model
        vector = self.embedding_client.embed_query(text_query)

        # TODO: use the vector representation of the text_query to
        # search the database by using the query function.
        result = self.index.query(
            namespace=None,
            vector=vector,
            top_k=top_k,
            include_values=True,
            include_metadata=True,
            filter=search_filter,
        )

        docs = []
        for res in result["matches"]:
            # TODO: From the result's metadata, extract the "text" element.
            docs.append(res["metadata"]["text"])

        return docs


if __name__ == "__main__":

    # from langchain_community.document_loaders import GitLoader
    # from langchain_text_splitters import (
    #     Language,
    #     RecursiveCharacterTextSplitter,
    # )

    # loader = GitLoader(
    #     clone_url="https://github.com/gulmert89/vault",
    #     repo_path=os.path.join(current_dir, "./code_data/the_github_repo/"),
    #     branch="main",
    # )

    # python_splitter = RecursiveCharacterTextSplitter.from_language(
    #     language=Language.PYTHON, chunk_size=5000, chunk_overlap=250
    # )

    # docs = loader.load()
    # docs = [doc for doc in docs if doc.metadata["file_type"] in [".py", ".md"]]
    # docs = [doc for doc in docs if len(doc.page_content) < 40000]
    # docs = python_splitter.split_documents(docs)
    # for doc in docs:
    #     doc.page_content = "# {}\n\n".format(doc.metadata["source"]) + doc.page_content

    # indexer = DataIndexer()
    # with open(os.path.join(current_dir, "sources.txt"), "a") as file:
    #     for doc in docs:
    #         file.writelines(doc.metadata["source"] + "\n")
    # indexer.index_data(docs)


    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    loader = PyPDFLoader(file_path="/home/mert/Documents/Mert Gul - CV.pdf")

    doc_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )

    docs = loader.load_and_split(
            text_splitter=doc_splitter
        )
    for doc in docs:
        doc.page_content = "# {}\n\n".format(doc.metadata["source"]) + doc.page_content

    indexer = DataIndexer(index_name="my-resume")
    with open(os.path.join(current_dir, "sources.txt"), "a") as file:
        for doc in docs:
            file.writelines(doc.metadata["source"] + "\n")
    indexer.index_data(docs)

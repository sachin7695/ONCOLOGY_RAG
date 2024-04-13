import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import SentenceTransformerEmbeddings

embeddings = SentenceTransformerEmbeddings(model_name = "neuml/pubmedbert-base-embeddings")
print(embeddings)

loader = DirectoryLoader('Data/', glob = "**/*.pdf", show_progress = True, loader_cls = PyPDFLoader)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 10
)
texts = text_splitter.split_documents(documents)

url = "http://localhost:6333/dashboard"

qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    url = url,
    prefer_grpc = False,
    collection_name = "vector_db"
)
print("Vector DB Successfully Created!")
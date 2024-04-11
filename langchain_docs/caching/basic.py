from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
import time

underlying_embeddings = OpenAIEmbeddings()

store = LocalFileStore("./cache/")

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings, store, namespace=underlying_embeddings.model)

# The cache is empty prior to embedding:
print(list(store.yield_keys()))

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
raw_documents = TextLoader("../../text_files/state_of_the_union.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

start = time.perf_counter()
db = FAISS.from_documents(documents, cached_embedder)
end = time.perf_counter()
print(f"Time to build vector store: {end - start}")

start = time.perf_counter()
db2 = FAISS.from_documents(documents, cached_embedder)
end = time.perf_counter()
print(f"Time to build vector store: {end - start}")

print(list(store.yield_keys())[:5])
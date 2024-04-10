from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore

from dotenv import load_dotenv
load_dotenv()
import os
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

loaders = [
    # TextLoader("../../text_files/paul_graham_essays.txt"),
    TextLoader("../../text_files/state_of_the_union.txt"),
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

# print(docs[0])

fs = LocalFileStore("./cache/")

# Retrieving larger chunks
# Sometimes, the full documents can be too big to want to retrieve them as is. In that case, what we really want to do is to first split the raw documents into larger chunks, and then split it into smaller chunks. We then index the smaller chunks, but on retrieval we retrieve the larger chunks (but still not the full documents).
# This text splitter is used to create the parent documents
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
# This text splitter is used to create the child documents
# It should create documents smaller than the parent
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
# The vectorstore to use to index the child chunks
vectorstore = Chroma(
    collection_name="split_parents", embedding_function=OpenAIEmbeddings(), persist_directory="./db"
)
# The storage layer for the parent documents
# store = InMemoryStore()
store = create_kv_docstore(fs)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# retriever.add_documents(docs)

# We can see that there are much more than two documents now - these are the larger chunks.
res = len(list(store.yield_keys()))
print(f"keys: {res}")

sub_docs = vectorstore.similarity_search("justice breyer")
# print(sub_docs) # [Document(page_content='...', metadata={'doc_id': 'f4919d92-51df-4418-bd83-41f6afd7390f', 'source': './text_files/state_of_the_union.txt'})]
print(sub_docs[0].page_content)

retrieved_docs = retriever.get_relevant_documents("justice breyer")
res = len(retrieved_docs[0].page_content)
print(f"len: {res}")
print(retrieved_docs[0].page_content)
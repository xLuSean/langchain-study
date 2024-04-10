from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
load_dotenv()
import os
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

loaders = [
    TextLoader("../../text_files/paul_graham_essays.txt"),
    TextLoader("../../text_files/state_of_the_union.txt"),
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

# print(docs[0])

#=== Retrieving full documents
# In this mode, we want to retrieve the full documents. Therefore, we only specify a child splitter.
# This text splitter is used to create the child documents
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
# The vectorstore to use to index the child chunks
vectorstore = Chroma(
    collection_name="full_documents", embedding_function=OpenAIEmbeddings()
)
# The storage layer for the parent documents
store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
)

retriever.add_documents(docs, ids=None)

# This should yield two keys, because we added two documents.
print(list(store.yield_keys()))

# Let’s now call the vector store search functionality - we should see that it returns small chunks (since we’re storing the small chunks).
sub_docs = vectorstore.similarity_search("justice breyer")
print(sub_docs[0].page_content)

# Let’s now retrieve from the overall retriever. This should return large documents - since it returns the documents where the smaller chunks are located.
retrieved_docs = retriever.get_relevant_documents("justice breyer")
print(len(retrieved_docs[0].page_content))
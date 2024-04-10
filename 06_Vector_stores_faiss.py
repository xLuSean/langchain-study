from dotenv import load_dotenv
load_dotenv()
import os
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
# ================================================================================

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
raw_documents = TextLoader('./text_files/state_of_the_union.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db = FAISS.from_documents(documents, OpenAIEmbeddings())

query = "What did the president say about Ketanji Brown Jackson?"

## Similarity search
# docs = db.similarity_search(query)
# print(docs[0].page_content)

## similarity search with score
# docs_and_score = db.similarity_search_with_score(query)
# print(docs_and_score[0])

## Similarity search by vector
# embedding_vector = OpenAIEmbeddings().embed_query(query)
# docs = db.similarity_search_by_vector(embedding_vector)
# print(docs[0].page_content)

## save and load
db.save_local("faiss_index")
new_db = FAISS.load_local("faiss_index", OpenAIEmbeddings(),allow_dangerous_deserialization=True)
docs = new_db.similarity_search(query)
print(docs[0].page_content)
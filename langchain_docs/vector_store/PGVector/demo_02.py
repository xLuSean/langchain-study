from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

loader = TextLoader("../../../text_files/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# See docker command above to launch a postgres instance with pgvector enabled.
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"  # Uses psycopg3!
collection_name = "my_docs"


db = PGVector.from_documents(
    embedding=embeddings,
    documents=docs,
    collection_name=collection_name,
    connection=connection,
)

# query = "What did the president say about Ketanji Brown Jackson"

# # # result = db.similarity_search_with_score(query, k=3)
# # # print(result)

# retriever = db.as_retriever(search_kwargs={"k": 3})
# res = retriever.invoke(query)
# print(res)
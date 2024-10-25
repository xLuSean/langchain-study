from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# See docker command above to launch a postgres instance with pgvector enabled.
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"  # Uses psycopg3!
collection_name = "demo_01"


vector_store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    # use_jsonb=True,
)

docs = [
    Document(page_content="there are cats in the pond",metadata={"id": 11, "location": "pond", "topic": "animals"},),
    Document(page_content="ducks are also found in the pond",metadata={"id": 12, "location": "pond", "topic": "animals"},),
    Document(page_content="fresh apples are available at the market",metadata={"id": 13, "location": "market", "topic": "food"},),
    Document(page_content="the market also sells fresh oranges",metadata={"id": 14, "location": "market", "topic": "food"},),
    Document(page_content="the new art exhibit is fascinating",metadata={"id": 15, "location": "museum", "topic": "art"},),
    Document(page_content="a sculpture exhibit is also at the museum",metadata={"id": 16, "location": "museum", "topic": "art"},),
    Document(page_content="a new coffee shop opened on Main Street",metadata={"id": 17, "location": "Main Street", "topic": "food"},),
    Document(page_content="the book club meets at the library",metadata={"id": 18, "location": "library", "topic": "reading"},),
    Document(page_content="the library hosts a weekly story time for kids",metadata={"id": 19, "location": "library", "topic": "reading"},),
    Document(page_content="a cooking class for beginners is offered at the community center",metadata={"id": 20, "location": "community center", "topic": "classes"},),
]

vector_store.add_documents(docs, ids=[doc.metadata["id"] for doc in docs])

result = vector_store.similarity_search(
    # "kitty", k=10, filter={"id": {"$in": [1, 5, 2, 9]}}
    "kitty", k=4
)

# result = vector_store.similarity_search_with_score("there are cats in the pond", k=3, filter={"location": "pond"})
print(result)
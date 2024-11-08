from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import SQLiteVec
from langchain_text_splitters import CharacterTextSplitter

# load the document and split it into chunks
loader = TextLoader("../../../text_files/state_of_the_union.txt")
documents = loader.load()

# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
texts = [doc.page_content for doc in docs]


# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
connection = SQLiteVec.create_connection(db_file="./tmp/vec.db")

db1 = SQLiteVec(
    table="state_union", embedding=embedding_function, connection=connection
)

for text in texts:
    db1.add_texts([text])

# query it again
query = "What did the president say about Ketanji Brown Jackson"
data = db1.similarity_search(query)

# print results
for row in data:
    print(row.page_content)
    print("=======================")
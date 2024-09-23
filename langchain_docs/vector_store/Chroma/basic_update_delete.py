# import
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores.chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter

# load the document and split it into chunks
document_path = "../../text_files/state_of_the_union.txt"
loader = TextLoader(document_path)
documents = loader.load()

# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# create a list of ids
ids = [str(i) for i in range(len(docs))]

# load it into Chroma
db = Chroma.from_documents(docs, embedding_function, ids=ids)

# query it
query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)
print(docs[0].metadata)

# update the metadata for a document
docs[0].metadata = {
    "source": "../../modules/state_of_the_union.txt",
    "new_value": "hello world",
}
db.update_document(ids[0], docs[0])
print(db._collection.get(ids=[ids[0]]))

# delete the last document
# print("count before", db._collection.count())
# db._collection.delete(ids=[ids[-1]])
# print("count after", db._collection.count())
# import
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores.chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter

#>>> load the document and split it into chunks
# loader = TextLoader("../../text_files/state_of_the_union.txt")
# documents = loader.load()

#>>> split it into chunks
# text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)

#>>> create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

#>>> save to disk
# db = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db")
#>>> load from disk
db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

#>>> In a notebook, we should call persist() to ensure the embeddings are written to disk. This isn't necessary in a script - the database will be automatically persisted when the client object is destroyed.
# db.persist()

#>>> query it
query = "What did the president say about Ketanji Brown Jackson"
# res = db.similarity_search(query, k=3)
res = db.similarity_search_with_score(query, k=3)

#>>> print results
# print(len(res))
# print(res[0].page_content)
for i in range(len(res)):
    print(f"### retreived document {i} with similarity ###\n {res[i]}")


#>>> When you're done with the database, you can delete it from disk. You can delete the specific collection you're working with (if you have several), or delete the entire database by nuking the persistence directory.

# To cleanup, you can delete the collection
# db.delete_collection()
# db.persist()
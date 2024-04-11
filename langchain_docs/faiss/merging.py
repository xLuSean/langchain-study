from langchain.vectorstores.faiss import FAISS
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings


embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

db1 = FAISS.from_texts(["foo"], embedding)
db2 = FAISS.from_texts(["bar"], embedding)
# 打印第一个FAISS
print(db1.docstore._dict)
print(db2.docstore._dict)

# 合并
db1.merge_from(db2)
# 打印
print(db1.docstore._dict)
print(db2.docstore._dict)

# print(db1.similarity_search("foo"))
print(db1.similarity_search_with_score("foo1", k=2))

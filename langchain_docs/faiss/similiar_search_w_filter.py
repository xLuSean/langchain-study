from langchain.schema import Document
from langchain.vectorstores.faiss import FAISS
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# 先构造文档数据，方便后面的测试
list_of_documents = [
    Document(page_content="foo", metadata=dict(page=1)),
    Document(page_content="bar", metadata=dict(page=1)),
    Document(page_content="foo", metadata=dict(page=2)),
    Document(page_content="barbar", metadata=dict(page=2)),
    Document(page_content="foo", metadata=dict(page=3)),
    Document(page_content="bar burr", metadata=dict(page=3)),
    Document(page_content="foo", metadata=dict(page=4)),
    Document(page_content="bar bruh", metadata=dict(page=4)),
]

db = FAISS.from_documents(list_of_documents, embedding)
#>>> 简单搜索下，方便后面的对比
# results_with_scores = db.similarity_search_with_score("foo")
# results_with_scores = db.similarity_search_with_score("foo", filter=dict(page=1)) # Filter by metadata

# for doc, score in results_with_scores:
#     print(f"Content: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")

#>>> max_marginal_relevance_search
# results = db.max_marginal_relevance_search("foo", filter=dict(page=1))
# for doc in results:
#     print(f"Content: {doc.page_content}, Metadata: {doc.metadata}")


#>>> 以下是调用similarity_search时如何设置 fetch_k 参数的示例。通常我们需要 fetch_k参数 >> k 参数。这是因为 fetch_k 参数是过滤之前将获取的文档数。如果将 fetch_k 设置为较小的数字，则可能无法获得足够的文档进行过滤。
# k设置过滤后得到的文档数、fetch_k设置过滤前的文档数
results = db.similarity_search("foo", filter=dict(page=1), k=1, fetch_k=4)
for doc in results:
    print(f"Content: {doc.page_content}, Metadata: {doc.metadata}")

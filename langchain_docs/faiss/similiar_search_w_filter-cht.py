# https://blog.csdn.net/u013066244/article/details/132014791

from langchain.schema import Document
from langchain.vectorstores.faiss import FAISS
# from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

# embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
embedding = HuggingFaceEmbeddings(model_name="thenlper/gte-large-zh", model_kwargs={'device': 'cpu'})

# 先构造文档数据，方便后面的测试
list_of_documents = [
    Document(page_content="我的寵物應該多久接受一次健康檢查？", metadata=dict(page=1)),
    Document(page_content="寵物疫苗應該怎麼安排？", metadata=dict(page=1)),
    Document(page_content="如果我的狗狗吃了巧克力怎麼辦？", metadata=dict(page=2)),
    Document(page_content="我的貓咪最近不吃飯，這正常嗎？", metadata=dict(page=2)),
    Document(page_content="你們診所可以做寵物絕育手術嗎？", metadata=dict(page=3)),
    Document(page_content="我應該如何處理我的寵物的牙齒問題？", metadata=dict(page=3)),
    Document(page_content="怎麼判斷我的寵物是否發燒了？", metadata=dict(page=4)),
    Document(page_content="我的寵物需要減肥，你有什麼建議？", metadata=dict(page=4)),
]


db = FAISS.from_documents(list_of_documents, embedding)
#>>> 简单搜索下，方便后面的对比
# results_with_scores = db.similarity_search_with_score("我的寵物應該多久接受一次健康檢查？")
results_with_scores = db.similarity_search_with_score("我的寵物應該什麼時候接受牙齒檢查？")
# results_with_scores = db.similarity_search_with_score("foo", filter=dict(page=1)) # Filter by metadata
# print(results_with_scores)

for doc, score in results_with_scores:
    if score < 0.5:
        print(f"Content: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")

#>>> max_marginal_relevance_search
# results = db.max_marginal_relevance_search("foo", filter=dict(page=1))
# for doc in results:
#     print(f"Content: {doc.page_content}, Metadata: {doc.metadata}")


#>>> 以下是调用similarity_search时如何设置 fetch_k 参数的示例。通常我们需要 fetch_k参数 >> k 参数。这是因为 fetch_k 参数是过滤之前将获取的文档数。如果将 fetch_k 设置为较小的数字，则可能无法获得足够的文档进行过滤。
# k设置过滤后得到的文档数、fetch_k设置过滤前的文档数
# results = db.similarity_search("foo", filter=dict(page=1), k=1, fetch_k=4)
# for doc in results:
#     print(f"Content: {doc.page_content}, Metadata: {doc.metadata}")

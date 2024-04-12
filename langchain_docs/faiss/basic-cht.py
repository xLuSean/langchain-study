# https://blog.csdn.net/u013066244/article/details/132014791
from dotenv import load_dotenv
load_dotenv()

#>>> Uncomment the following line if you need to initialize FAISS with no AVX2 optimization
#>>> 如果您需要在没有 AVX2 优化的情况下初始化 FAISS，请取消以下注释
# os.environ['FAISS_NO_AVX2'] = '1'

# from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain_community.document_loaders import TextLoader

loader = TextLoader("../../text_files/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

#>>> embeddings = OpenAIEmbeddings()
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

db = FAISS.from_documents(docs, embedding)

query = "What did the president say about Ketanji Brown Jackson"

# >>> basic search
# docs = db.similarity_search(query)
# print(docs[0].page_content)

#>>> 有一些 FAISS 特定方法。其中之一是similarity_search_with_score，它不仅允许您返回文档，还允许返回查询到它们的距离分数。返回的距离分数是L2距离。因此，分数越低越好。

docs_and_scores = db.similarity_search_with_score(query, k=4) # default k=4
docs_and_scores[1][0].metadata.update({"test": "test"})
print(docs_and_scores[1][0])  # 1st element is the document, 2nd element is the score
print(len(docs_and_scores[1]))

#>>> 还可以使用similarity_search_by_vector 与给定嵌入向量相似的文档进行搜索，该向量接受嵌入向量作为参数而不是字符串
# embed 向量
# embedding_vector = embedding.embed_query(query)
# docs = db.similarity_search_by_vector(embedding_vector)
# print(len(docs))

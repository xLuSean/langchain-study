# https://blog.csdn.net/u013066244/article/details/132014791

from dotenv import load_dotenv
load_dotenv()

#>>> Uncomment the following line if you need to initialize FAISS with no AVX2 optimization
#>>> 如果您需要在没有 AVX2 优化的情况下初始化 FAISS，请取消以下注释
# os.environ['FAISS_NO_AVX2'] = '1'

# from langchain_openai import OpenAIEmbeddings
# from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores.faiss import FAISS
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader

loader = TextLoader("../../../text_files/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# #>>> embeddings = OpenAIEmbeddings()
# embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# >>> saving >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# db = FAISS.from_documents(docs, embedding)
# db.save_local("faiss_index")
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# >>> loading >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
new_db = FAISS.load_local("faiss_index", embeddings=embedding, allow_dangerous_deserialization=True)
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

#>>> 保存和加载（Saving and loading）您还可以保存和加载 FAISS 索引。这很有用，因此我们不必每次使用它时都重新创建它。


query = "What did the president say about Ketanji Brown Jackson"


docs = new_db.similarity_search(query)
print(docs[0].page_content)
# print(docs)

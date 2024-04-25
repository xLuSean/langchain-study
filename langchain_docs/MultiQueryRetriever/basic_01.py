#>>> Build a sample vectorDB
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

#>>> Load blog post
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

#>>> Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
splits = text_splitter.split_documents(data)

#>>> VectorDB
embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents=splits, embedding=embedding)

#>>> Simple usage
#>>> Specify the LLM to use for query generation, and the retriever will do the rest.
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

question = "What are the approaches to Task Decomposition?"

llm = ChatOpenAI(temperature=0)

retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(), llm=llm)

#>>> Set logging for the queries
import logging
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

unique_docs = retriever_from_llm.get_relevant_documents(query=question)
# print(len(unique_docs))
# print(f"#####\n {unique_docs}")
# for doc in unique_docs:
#     print(doc.page_content)
#     print("#####")

documents = "###".join(doc.page_content for doc in unique_docs)
print(documents)

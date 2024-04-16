from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os


load_dotenv()
print(os.getenv("LANGCHAIN_API_KEY"))

llm = Ollama(model='llama2')
embeddings = OllamaEmbeddings()

loader = WebBaseLoader("https://edition.cnn.com/2024/03/06/tech/openai-elon-musk-emails/index.html")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter()
docs = text_splitter.split_documents(documents)

vector = FAISS.from_documents(docs, embeddings)
retriever = vector.as_retriever()

prompt = ChatPromptTemplate.from_messages([
    ('system', 'Answer the user\'s questions based on the below context:\n\n{context}'),
    ('user', 'Question: {input}'),
])
document_chain = create_stuff_documents_chain(llm, prompt)

retrieval_chain = create_retrieval_chain(retriever, document_chain)

context = []
input_text = input('>>> ')
while input_text.lower() != 'bye':
    response = retrieval_chain.invoke({
        'input': input_text,
        'context': context
    })
    print(response['answer'])
    context = response['context']
    input_text = input('>>> ')
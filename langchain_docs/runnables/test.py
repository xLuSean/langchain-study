from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate


llm = Ollama(model='llama2', temperature=0.9)
prompt = ChatPromptTemplate.from_messages([
    ("user", "{input}"),
])
chain = prompt | llm
print(chain.invoke({'input': 'Tell me a joke'}))
from langchain_community.llms.ollama import Ollama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import ConfigurableField

llm = Ollama(model='llama2').configurable_alternatives(
    ConfigurableField(id="llm"),
    default_key='llama2',
    gpt35=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    gpt4=ChatOpenAI(model="gpt-4-turbo", temperature=0),
)

prompt = ChatPromptTemplate.from_messages([
    ("user", "{input}"),
])
chain = prompt | llm

# print(chain.with_config(configurable={"llm": "gpt35"}).invoke({'input': 'Tell me a joke'}))
print(chain.with_config(configurable={"llm": "gpt4"}).invoke({'input': 'Tell me a joke'}))
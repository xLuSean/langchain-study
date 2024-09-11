# we are dynamically changin the temperature and LLM model in runtime
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import ConfigurableField


llm = Ollama(model='llama3.1', temperature=0).configurable_fields(
    temperature=ConfigurableField(
        id="temperature",
        name="LLM Temperature",
        description="The temperature of the LLM",
    ),
    model=ConfigurableField(
        id="model",
        name="The Model",
        description="The language model",
    ),
)

prompt = ChatPromptTemplate.from_messages([
    ("user", "{input}"),
])
chain = prompt | llm

print(chain.with_config(
    configurable={
        "model": "llama",
        "temperature": 0.9
    }
).invoke({'input': 'Tell me a joke'}))
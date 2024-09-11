from langchain_community.llms.ollama import Ollama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import ConfigurableField

llm = Ollama(model='llama2').configurable_alternatives(
    ConfigurableField(id="llm"),
    default_key='llama2',
    gpt35=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    gpt4=ChatOpenAI(model="gpt-4-turbo", temperature=0),
)

prompt = PromptTemplate.from_template(
    "Tell me a joke about {topic}"
).configurable_alternatives(
    # This gives this field an id
    # When configuring the end runnable, we can then use this id to configure this field
    ConfigurableField(id="prompt"),
    # This sets a default_key.
    # If we specify this key, the default LLM (ChatAnthropic initialized above) will be used
    default_key="joke",
    # This adds a new option, with name `poem`
    poem=PromptTemplate.from_template("Write a short poem about {topic}"),
    # You can add more configuration options here
)

chain = prompt | llm

# res = chain.invoke({'topic': 'cats'})
# print(res)

# res = chain.with_config(configurable={"prompt":"poem"}).invoke({'topic': 'cats'})
# print(res)

# print(chain.with_config(configurable={"llm": "gpt35"}).invoke({'input': 'Tell me a joke'}))
# print(chain.with_config(configurable={"llm": "gpt4"}).invoke({'input': 'Tell me a joke'}))

# Saving configurations
openai_gpt4 = chain.with_config(configurable={"llm":"gpt4"})
res = openai_gpt4.invoke({'topic': 'box'})
print(res)

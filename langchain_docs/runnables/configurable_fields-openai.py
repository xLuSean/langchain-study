from langchain.prompts import PromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI

model = ChatOpenAI(temperature=0).configurable_fields(
     model_name=ConfigurableField(
         id="model_name",
         name="The Model",
         description="The language model",
     ),
    temperature=ConfigurableField(
        id="llm_temperature",
        name="LLM Temperature",
        description="The temperature of the LLM",
    )
)

# res = model.invoke("pick a random number")
# print(res)

res = model.with_config(configurable={
    "model_name":"gpt-4-turbo-2024-04-09",
    "llm_temperature": 0.9}
    ).invoke("pick a random number between 0 and 1024")
print(res)
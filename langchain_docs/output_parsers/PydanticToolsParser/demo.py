from typing import List, Literal

from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import PromptTemplate
# from langchain_core.pydantic_v1 import BaseModel, Field, validator
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


# Set up a Pydantic model for the structured output

class Entity(BaseModel):
    name: str = Field(description="name of the entity")
    label: Literal["PERSON", "ORGANIZATION", "LOCATION"]


class ExtractEntities(BaseModel):
    entities: List[Entity]


# Choose a model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

prompt = PromptTemplate(
    template="""Distill the entities from the user's input, if no available return empty.
            If not sure, don't enforece
            User:{query}""",
            input_variables=["query"],
            )

# Force the model to always use the ExtractEntities schema
llm_with_tools = llm.bind_tools([ExtractEntities], tool_choice="ExtractEntities")

# Add a parser to convert the LLM output to a Pydantic object
chain = prompt | llm_with_tools | PydanticToolsParser(tools=[ExtractEntities])

text = """BioNTech SE is set to acquire InstaDeep, \
a Tunis-born and U.K.-based artificial intelligence \
(AI) startup, for up to £562 million\
"""
res = chain.invoke({"query": text})
print(type(res))
print(res)

for entity in res[0].entities:
    print(f"{entity.name}: {entity.label}")
from dotenv import load_dotenv
load_dotenv()
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

from langchain.agents import AgentExecutor, ZeroShotAgent
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities.serpapi import SerpAPIWrapper

from langchain_openai import OpenAI

search = SerpAPIWrapper()

from langchain.agents import Tool
tools = [
    Tool(
    name="search",
    func=search.run,
    description="useful when you need to answer question about updated knowledge"
)
]


prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)
memory = ConversationBufferMemory(memory_key="chat_history")

llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)

agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)

agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory
)

res = agent_chain.invoke({"input":"How many people live in canada?"})

print(res)

res = agent_chain.invoke(input="what is their national anthem called?")

print(res)
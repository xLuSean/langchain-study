# ref:
# https://python.langchain.com/docs/modules/memory/agent_with_memory_in_db

from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
# from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_community.utilities.serpapi import SerpAPIWrapper
from langchain_openai import OpenAI

search = SerpAPIWrapper()

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


message_history = RedisChatMessageHistory(
    url="redis://localhost:6379/0", ttl=600, session_id="my-session"
)

memory = ConversationBufferMemory(
    memory_key="chat_history", chat_memory=message_history
)

llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory
)

agent_chain.run(input="How many people live in canada?")
agent_chain.run(input="what is their national anthem called?")
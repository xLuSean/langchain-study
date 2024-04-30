from langchain import hub
from langchain.agents import AgentExecutor, Tool
from langchain.agents.react.agent import create_react_agent
from langchain_community.utilities.serpapi import SerpAPIWrapper
from langchain_openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

search = SerpAPIWrapper()

tools = [
    Tool(
    name="search",
    func=search.run,
    description="useful when you need to answer question about updated knowledge, only use if needed."
)
]

# # Choose the LLM to use
llm = OpenAI()

# >>> basic usage >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# # Get the prompt to use - you can modify this!
# prompt = hub.pull("hwchase17/react")

# # Construct the ReAct agent
# agent = create_react_agent(llm, tools, prompt)

# # Create an agent executor by passing in the agent and tools
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# res = agent_executor.invoke({"input": "what is LangChain?"})
# print(res)

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# >>> useage with chat history >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/react-chat")

# Construct the ReAct agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

res = agent_executor.invoke(
    {
        "input": "what's my name? Only use a tool if needed, otherwise respond with Final Answer",
        # Notice that chat_history is a string, since this prompt is aimed at LLMs, not chat models
        "chat_history": "Human: Hi! My name is Bob\nAI: Hello Bob! Nice to meet you",
    }
)

# res = agent_executor.invoke([
#     HumanMessage(content="what's my name? Only use a tool if needed, otherwise respond with Final Answer"),
#     SystemMessage(content="Human: Hi! My name is Bob\nAI: Hello Bob! Nice to meet you")
#     ])


print(res)

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# from getpass import getpass
import os
from api_keys import OPENAI_API_KEY, SERPAPI_API_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["SERPAPI_API_KEY"] = SERPAPI_API_KEY

# === key setting ==============================================================

from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
# from langchain.tools import BaseTool
from langchain_openai import OpenAI
from langchain.chains import LLMMathChain
from langchain_community.utilities import SerpAPIWrapper

llm = OpenAI(temperature=0)

# 初始化搜索链和计算链
search = SerpAPIWrapper()
llm_math_chain = LLMMathChain(llm=llm, verbose=True)

# 创建一个功能列表，指明这个 agent 里面都有哪些可用工具，agent 执行过程可以看必知概念里的 Agent 那张图
tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math"
    )
]

# 初始化 agent
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# 执行 agent
# agent.invoke("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?")
# agent.run("what is the age of universe? what is the age raised to the 0.5 power?")
agent.run("what is the most updated MSFT stock price? what is the price raised to the power of 0.5?")

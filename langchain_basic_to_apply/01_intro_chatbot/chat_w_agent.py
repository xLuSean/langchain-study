from dotenv import load_dotenv
load_dotenv()

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI


# 首先，我们加载我们要用来控制代理的语言模型
chat = ChatOpenAI(temperature=0)
 
# 其次，我们加载一些要使用的工具。请注意，“llm-math”工具使用LLM，所以我们需要传递它
llm = OpenAI(temperature=0)
# ImportError: LLMMathChain requires the numexpr package.
tools = load_tools(["serpapi", "llm-math"], llm=llm) # pip install numexpr
 
# 最后，我们使用工具、语言模型和我们要使用的代理类型来初始化代理
agent = initialize_agent(tools, chat, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
 
# 测试代理
res = agent.invoke({"Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?"})
# res = agent.invoke("Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?")

print(res)
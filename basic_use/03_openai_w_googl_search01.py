##############################################
### description: openai with serpapi to use google search in llm
##############################################
from dotenv import load_dotenv
load_dotenv()
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# === key setting ==============================================================

from langchain.agents import load_tools, initialize_agent, AgentType, Tool
from langchain_openai import OpenAI
from langchain_community.utilities.serpapi import SerpAPIWrapper

# 加载 OpenAI 模型
llm = OpenAI(temperature=0,max_tokens=2048) 

 # 加载 serpapi 工具
# >>> 1st method >>>>>>>>>>>>>>>>>>>>
# tool = load_tools(["serpapi"])
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# >>> 2nd method >>>>>>>>>>>>>>>>>>>>
search = SerpAPIWrapper()
tool = [
    Tool(
    name="search",
    func=search.run,
    description="useful when you need to answer question about updated knowledge"
)
]
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# 如果搜索完想再计算一下可以这么写
# tools = load_tools(['serpapi', 'llm-math'], llm=llm)

# 如果搜索完想再让他再用python的print做点简单的计算，可以这样写
# tools=load_tools(["serpapi","python_repl"])

# 工具加载后都需要初始化，verbose 参数为 True，会打印全部的执行详情
agent = initialize_agent(tool, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
# 关于agent type 几个选项的含义（理解不了也不会影响下面的学习，用多了自然理解了）：
# zero-shot-react-description: 根据工具的描述和请求内容的来决定使用哪个工具（最常用）
# react-docstore: 使用 ReAct 框架和 docstore 交互, 使用Search 和Lookup 工具, 前者用来搜, 后者寻找term, 举例: Wipipedia 工具
# self-ask-with-search 此代理只使用一个工具: Intermediate Answer, 它会为问题寻找事实答案(指的非 gpt 生成的答案, 而是在网络中,文本中已存在的), 如 Google search API 工具
# conversational-react-description: 为会话设置而设计的代理, 它的prompt会被设计的具有会话性, 且还是会使用 ReAct 框架来决定使用来个工具, 并且将过往的会话交互存入内存


# 运行 agent
# query = "What's the date today? What great events have taken place today in history?"
query = "台灣貓糧推薦排行榜，用中文回覆"
# query = "我的10歲狗狗，早上吃乾糧的時候吐了怎麼辦？"
res = agent.invoke(query)

print(res)
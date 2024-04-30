# TODO: not working, need to fix
# Ref: https://wangwei1237.github.io/LLM_in_Action/langchain_agent_react.html
from langchain_openai import OpenAI, ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

# llm = OpenAI(model="gpt-3.5-turbo")
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

# from langchain.chains import LLMMathChain
from langchain.chains.llm_math.base import LLMMathChain
from langchain.agents import Tool

from langchain_core.prompts.chat import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("user", "你是一个能力非凡的人工智能机器人。"),
    ("assistant", "你好~"),
    ("user", "{user_input}"),
])

from langchain.chains.llm import LLMChain
llm_chain = LLMChain(llm=llm, prompt=template)

# llm_math = LLMMathChain(llm=llm) # 直接在构造函数中通过 llm 参数来初始化 Tool 的方式已经不再推荐使用。
llm_math = LLMMathChain.from_llm(llm)

# initialize the math tool
# 在初始化工具时，要特别注意对 description 属性的赋值。因为 Agent 主要根据该属性值来判断接下来将要采用哪个工具来执行后续的操作。优秀的 description 有利于最终任务的完美解决。
math_tool = Tool(
    name='Calculator',
    func=llm_math.run,
    description='Useful for when you need to answer questions about math.'
)

# initialize the general LLM tool
llm_tool = Tool(
    name='Language Model',
    func=llm_chain.run,
    description='Use this tool for general purpose queries.'
)


# when giving tools to LLM, we must pass as list of tools
tools = [math_tool, llm_tool]

# from langchain.agents import load_tools
# tools = load_tools(
#     ['llm-math'], # equalt to _get_llm_math, which is same as we defined above
#     llm=llm
# )

# get the prompt template string from: 
# https://smith.langchain.com/hub/hwchase17/react?organizationId=c4887cc4-1275-5361-82f2-b22aee75bad1

from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain.agents.react.agent import create_react_agent
from langchain.agents import AgentExecutor
prompt = hub.pull("hwchase17/react-chat")
# prompt_template = """..."""
# prompt = PromptTemplate.from_template(prompt_template)

zero_shot_agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

agent_executor = AgentExecutor(agent=zero_shot_agent, tools=tools, verbose=True)
# try:
#     res = agent_executor.invoke({"input": "what's 4.1*7.9=?"})
# except Exception as e:
#     res = {}

res = agent_executor.invoke({"input": "what's 4.1*7.9=?", "chat_history": ""})

print(res)
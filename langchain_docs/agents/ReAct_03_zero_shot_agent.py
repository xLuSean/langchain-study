from langchain_openai import OpenAI, ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.llm_math.base import LLMMathChain
from langchain.agents import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub

from dotenv import load_dotenv
load_dotenv()

# llm = OpenAI(model="gpt-3.5-turbo") # can't use this
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# initialize the math tool
llm_math = LLMMathChain.from_llm(llm)
math_tool = Tool(
    name='Calculator',
    func=llm_math.run,
    description='Useful for when you need to answer questions about math.'
)

# initialize the general LLM tool
template = ChatPromptTemplate.from_messages([
    ("user", "你是一个能力非凡的人工智能机器人。"),
    ("assistant", "你好~"),
    ("user", "{user_input}"),
])
llm_chain = LLMChain(llm=llm, prompt=template)
llm_tool = Tool(
    name='Language Model',
    func=llm_chain.run,
    description='Use this tool for general purpose queries.'
)

# when giving tools to LLM, we must pass as list of tools
tools = [math_tool, llm_tool]

# get the prompt template string from: 
# https://smith.langchain.com/hub/hwchase17/react?organizationId=c4887cc4-1275-5361-82f2-b22aee75bad1
# prompt_template = """..."""
# prompt = PromptTemplate.from_template(prompt_template)

prompt = hub.pull("hwchase17/react-chat")

zero_shot_agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

# >>> show the current prompting >>>
prompt_template = zero_shot_agent.get_prompts()[0]
# print(prompt_template)
print(prompt_template.format(input="what's 4.1*7.9=?", agent_scratchpad="", chat_history=""))
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

agent_executor = AgentExecutor(agent=zero_shot_agent, tools=tools, verbose=True)
try:
    # res = agent_executor.invoke({"input": "what's 4.1*7.9=?", "chat_history": ""})
    res = agent_executor.invoke({"input": "what's the capital of China?", "chat_history": ""})
except Exception as e:
    res = {}

# res = agent_executor.invoke({"input": "what's 4.1*7.9=?", "chat_history": ""})

print(res)
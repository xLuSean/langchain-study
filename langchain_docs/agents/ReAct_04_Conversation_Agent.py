# https://wangwei1237.github.io/LLM_in_Action/langchain_agent_react.html
from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.llm_math.base import LLMMathChain
from langchain.agents import Tool
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub

from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4", temperature=0)

llm_math = LLMMathChain.from_llm(llm)

template = ChatPromptTemplate.from_messages([
    ("user", "你是一个能力非凡的人工智能机器人。"),
    ("assistant", "你好~"),
    ("user", "{user_input}"),
])
llm_chain = LLMChain(llm=llm, prompt=template)

# initialize the math tool
math_tool = Tool(
    name='Calculator',
    func=llm_math.run,
    description='Useful for when you need to answer questions about math.'
)

# initialize the general LLM tool
llm_tool = Tool(
    name='Language Model',
    func=llm_chain.run,
    description='Use this tool for general purpose queries. Answer in Chinese'
)

# search Tool
search = DuckDuckGoSearchRun()
search_tool = Tool(
    name='search',
    func=search.run,
    description='useful when you need to answer question about updated knowledge, only use if needed.'
)

# when giving tools to LLM, we must pass as list of tools
tools = [math_tool, llm_tool, search_tool]

# get the prompt template string from: 
# https://smith.langchain.com/hub/hwchase17/react-chat?organizationId=c4887cc4-1275-5361-82f2-b22aee75bad1

# prompt = hub.pull("hwchase17/react-chat")

prompt_template = """Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

Unfortunately, Assistant is terrible at maths. When provided with math questions, no matter how simple, assistant always refers to it's trusty tools and absolutely does NOT try to answer math questions by itself.

TOOLS:
------

Assistant has access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}"""
prompt = PromptTemplate.from_template(prompt_template)

conversation_agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)



history = ["Human: 今年是哪一年？，AI: 今年是西元1768年。"]
querys = [
    "这一年，中國有什么重大事件发生？",
    "同年，其他国家有什么重大事件发生？",
    "3.1415926*4 是多少？"
]

# >>> show the current prompting >>>
# prompt_template = conversation_agent.get_prompts()[0]
# # print(prompt_template)
# print(prompt_template.format(input=querys[0], agent_scratchpad="", chat_history=history))
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

agent_executor = AgentExecutor(agent=conversation_agent, tools=tools, verbose=True)

for query in querys:
    try:
        res = agent_executor.invoke({"input": query, "chat_history": "\n".join(history)})
    except Exception as e:
        res = {}

    history.append("Human: " + query + "\nAI: " + res.get("output", ""))

    print(res)
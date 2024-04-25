from dotenv import load_dotenv
load_dotenv()
from langchain_openai import OpenAI, ChatOpenAI
# from langchain.callbacks import get_openai_callback
from langchain_community.callbacks import get_openai_callback
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

llm = OpenAI()
chat = ChatOpenAI()

#=== basic use
# with get_openai_callback() as cb:
#     result = llm.invoke("Tell me a joke")
#     print(cb)

#=== basic use --- with multiple invokes
# with get_openai_callback() as cb:
#     result = llm.invoke("Tell me a joke")
#     result2 = llm.invoke("Tell me a joke")
#     print(cb.total_cost)
#     print(cb.total_tokens)

#=== basic use --- with multiple steps
with get_openai_callback() as cb:
    tools = load_tools(["serpapi", "llm-math"], llm=llm) # pip install numexpr
    agent = initialize_agent(tools, chat, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    res = agent.invoke({"Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?"})
    print(res)
    # print(cb.total_cost)
    print(cb)
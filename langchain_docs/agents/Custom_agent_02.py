from dotenv import load_dotenv
load_dotenv()
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#################################################
### Load the LLM
#################################################

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

#################################################
### Define Tools
#################################################
from langchain.agents import tool
@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

@tool
def do_math(expr) -> int:
    """add two number and return."""
    return eval(expr)

# res = get_word_length.invoke("abc d")
# print(res)

# res = do_math.invoke("1+2+3+4+5+6+7+8+9+10")
# print(res)

tools = [get_word_length, do_math]

################################################
### Create Prompt & Adding memory
################################################

from langchain_core.prompts import ChatPromptTemplate#, MessagesPlaceholder
from langchain.prompts import MessagesPlaceholder

# Add a place for memory variables to go in the prompt
# Keep track of the chat history

MEMORY_KEY = "chat_history"
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are very powerful assistant, but bad at calculating lengths of words.",
        ),
        MessagesPlaceholder(variable_name=MEMORY_KEY),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

from langchain_core.messages import AIMessage, HumanMessage

chat_history = []

####################################################
### Bind tools to LLM
####################################################

# How does the agent know what tools it can use? In this case we’re relying on OpenAI tool calling LLMs, which take tools as a separate argument and have been specifically trained to know when to invoke those tools.

llm_with_tools = llm.bind_tools(tools)

###############################################################
### Create the Agent
###############################################################
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

input1 = "how many letters in the word educa?"
result = agent_executor.invoke({"input": input1, "chat_history": chat_history})
chat_history.extend(
    [
        HumanMessage(content=input1),
        AIMessage(content=result["output"]),
    ]
)
agent_executor.invoke({"input": "is that a real word?", "chat_history": chat_history})



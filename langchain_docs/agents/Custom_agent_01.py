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
### Create Prompt
################################################

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# input should be a string containing the user objective. agent_scratchpad should be a sequence of messages that contains the previous agent tool invocations and the corresponding tool outputs.
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are very powerful assistant, but don't know current events",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

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
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

res = list(agent_executor.stream({"input": "How many letters in the word eudca"}))

print(res)

# res_alone = llm.invoke("How many letters in the word educa")
# print(f"Alone: {res_alone}")



from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.prompts.chat import (
    AIMessage,
    ChatPromptTemplate,
    HumanMessage,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.prompts import PromptTemplate
from langchain import hub

from dotenv import load_dotenv
load_dotenv()

# from tools import PythagorasTool
# from tools import CircumferenceTool
from typing import Union, Optional
from math import pi, sqrt, cos, sin
from langchain.tools import BaseTool

class CircumferenceTool(BaseTool):
    name = "Circumference calculator"
    description = """use this tool when you need to calculate
      a circumference using the radius of a circle"""
    def _run(self, radius: Union[int, float]):
        return float(radius)*2.0*pi
    def _arun(self, radius: Union[int, float]):
        raise NotImplementedError("This tool does not support asynchronous execution")

desc = (
    "use this tool when you need to calculate the length of an hypotenuse "
    "given one or two sides of a triangle and/or an angle (in degrees). "
    "To use the tool you must provide at least two of the following parameters "
    "['adjacent_side', 'opposite_side', 'angle']."
)
class PythagorasTool(BaseTool):
    name = "Hypotenuse calculator"
    description = desc
    def _run(self, adjacent_side: Optional[Union[int, float]] = None,
             opposite_side: Optional[Union[int, float]] = None,
             angle: Optional[Union[int, float]] = None):
        # check if at least two parameters are provided
        if adjacent_side and opposite_side:
            return sqrt(float(adjacent_side)**2 + float(opposite_side)**2)
        elif adjacent_side and angle:
            return adjacent_side / cos(float(angle))
        elif opposite_side and angle:
            return opposite_side / sin(float(angle))
        else:
            return "Could not calculate the hypotenuse. Please provide at least two of the following parameters: ['adjacent_side', 'opposite_side', 'angle']"

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support asynchronous execution")

# llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0)
# llm = ChatAnthropic(model_name="claude-3-haiku-20240307", temperature=0)
llm = ChatAnthropic(model_name="claude-3-sonnet-20240229", temperature=0)

# when giving tools to LLM, we must pass as list of tools
tools = [CircumferenceTool(), PythagorasTool()]

# the prompt template can get from: 
# https://smith.langchain.com/hub/hwchase17/structured-chat-agent?organizationId=c4887cc4-1275-5361-82f2-b22aee75bad1
# messages = hub.pull("hwchase17/structured-chat-agent")

system_message_template = """Respond to the human as helpfully and accurately as possible. You have access to the following tools:
{tools}
Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).
Valid "action" values: "Final Answer" or {tool_names}
Provide only ONE action per $JSON_BLOB, as shown:
```
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
```

Follow this format:
Question: input question to answer
Thought: consider previous and subsequent steps
Action:
```
$JSON_BLOB
```
Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action:
```
{{
  "action": "Final Answer",
  "action_input": "Final response to human"
}}

Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation

{agent_scratchpad}
 (reminder to respond in a JSON blob no matter what)"""
human_message_template = """{input}"""

messages = [
    SystemMessagePromptTemplate.from_template(system_message_template),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template(human_message_template),
]

input_variables = ["tools", "tool_names", "input", "chat_history", "agent_scratchpad"]
prompt = ChatPromptTemplate(input_variables=input_variables, messages=messages)

struct_agent = create_structured_chat_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)


agent_executor = AgentExecutor(agent=struct_agent, tools=tools, verbose=True)

history = []
querys = [
    """If I have a triangle with the opposite side of length 51 and the adjacent side of 40,
    what is the length of the hypotenuse?""",
]

res = agent_executor.invoke({"input": querys[0], "chat_history": history})
print(res)

# for query in querys:
#     try:
#         res = agent_executor.invoke({"input": query, "chat_history": history})
#     except Exception as e:
#         res = {}

#     history.append(HumanMessage(content=query))
#     history.append(AIMessage(content=res.get("output", "")))
    
#     print(res)
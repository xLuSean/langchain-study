from dotenv import load_dotenv
load_dotenv()
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

from langchain.tools import BaseTool
class Evaluate_math_tool(BaseTool):
    name = "calculator"
    description = "use this tool to evaluate a math expression"
    def _run(self, expr:str):
        return eval(expr)
    
tools = [Evaluate_math_tool()]

from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent

agent = initialize_agent(
    agent = "chat-conversational-react-description", # pick from preset
    tools = tools,
    llm = llm,
    verbose = True,
    max_iterations = 3,
    early_stopping_method = 'generate',
    memory = ConversationBufferWindowMemory(memory_key="chat_history", k=5, return_messages=True)
)

res = agent.invoke(f"what is 3*(3**2) ?")
# res = agent.invoke(f"if I have a cat and a duck, how many legs are there in total?")
print(res)
# res = agent.invoke(f"who is the president of the united states?")
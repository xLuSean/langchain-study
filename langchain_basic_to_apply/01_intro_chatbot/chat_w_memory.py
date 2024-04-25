from dotenv import load_dotenv
load_dotenv()

from langchain.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)
from langchain.chains.conversation.base import ConversationChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
 
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know."),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])
llm = ChatOpenAI(temperature=0)
memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)
# res = conversation.predict(input="Hi there!")
# print(res) 

# res = conversation.predict(input="I'm doing well! Just having a conversation with an AI.")
# print(res) 

# res = conversation.predict(input="Tell me about yourself.")
# print(res) 
while True:
    user_input = input("You: ")
    res = conversation.predict(input=user_input)
    print(res)

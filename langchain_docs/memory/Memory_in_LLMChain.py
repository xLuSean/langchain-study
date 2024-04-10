from dotenv import load_dotenv
load_dotenv()
import os
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
# ================================================================================

from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI


# >>> basic >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
## make sure the keys in the PromptTemplate and the ConversationBufferMemory match up (chat_history).

# template = """You are a chatbot having a conversation with a human.

# {chat_history}
# Human: {human_input}
# Chatbot:"""

# prompt = PromptTemplate(
#     input_variables=["chat_history", "human_input"], template=template
# )
# memory = ConversationBufferMemory(memory_key="chat_history")

# llm = OpenAI()
# llm_chain = LLMChain(
#     llm=llm,
#     prompt=prompt,
#     verbose=True,
#     memory=memory,
# )

# res = llm_chain.predict(human_input="Hi there my friend")
# print(res)
# res = llm_chain.predict(human_input="Not too bad - how are you?")
# print(res)

# <<< basic <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# The above works for completion-style LLMs, but if you are using a chat model, you will likely get better performance using structured chat messages. Below is an example.

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

## The configuration below makes it so the memory will be injected to the middle of the chat prompt, in the chat_history key, and the user’s inputs will be added in a human/user message to the end of the chat prompt.

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="You are a chatbot having a conversation with a human."
        ),  # The persistent system prompt
        MessagesPlaceholder(
            variable_name="chat_history"
        ),  # Where the memory will be stored.
        HumanMessagePromptTemplate.from_template(
            "{human_input}"
        ),  # Where the human input will injected
    ]
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

llm = ChatOpenAI()

chat_llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    # verbose=True,
    memory=memory,
)

res = chat_llm_chain.predict(human_input="Hi there my friend")
print(res)
res = chat_llm_chain.predict(human_input="Not too bad - how are you?")
print(res)
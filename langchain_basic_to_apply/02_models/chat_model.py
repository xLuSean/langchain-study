# 聊天模型是语言模型的一种变体。虽然聊天模型在内部使用语言模型，但它们公开的接口略有不同。它们不是提供一个“输入文本，输出文本”的API，而是提供一个以“聊天消息”作为输入和输出的接口。 聊天模型的API还比较新，因此我们仍在确定正确的抽象层次。本问将介绍如何开始使用聊天模型，该接口是基于消息而不是原始文本构建的

from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

chat = ChatOpenAI(temperature=0)
# res = chat.invoke([HumanMessage(content="Translate this sentence from English to French. I love programming.")])
# print(res)

# chat = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0)
# chat.invoke([HumanMessage(content="Translate this sentence from English to French. I love programming.")])

# messages = [
#     SystemMessage(content="You are a helpful assistant that translates English to French."),
#     HumanMessage(content="I love programming.")
# ]
# chat.invoke(messages)

batch_messages = [
    [
        SystemMessage(content="You are a helpful assistant that translates English to French."),
        HumanMessage(content="I love programming.")
    ],
    [
        SystemMessage(content="You are a helpful assistant that translates English to French."),
        HumanMessage(content="I love artificial intelligence.")
    ],
]
result = chat.generate(batch_messages)

print(result.llm_output)
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

# from langchain import PromptTemplate
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

chat = ChatOpenAI(temperature=0)

template="You are a helpful assistant that translates english to pirate."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
### option 1
# example_human = HumanMessagePromptTemplate.from_template("Hi")
# example_ai = AIMessagePromptTemplate.from_template("Argh me mateys")

### option 2, use additional_kwart
#### OpenAI提供了一个可选的name参数，我们也建议与系统消息一起使用以进行少量示例提示。以下是如何使用此功能的示例：
example_human = SystemMessagePromptTemplate.from_template("Hi", additional_kwart={"name":"example_user"})
example_ai = SystemMessagePromptTemplate.from_template("Argh me mateys", additional_kwart={"name":"example_assistant"})

human_template="{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, example_human, example_ai, human_message_prompt])

chain = LLMChain(llm=chat, prompt=chat_prompt)

# 从格式化的消息中获取聊天完成结果
# print(chain.invoke("I love programming."))
### or
# print(chain.invoke([HumanMessage(content="I love apple pie")]))

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
chat = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0)
resp = chat.invoke([HumanMessage(content="write a poem about hotdogs.")])
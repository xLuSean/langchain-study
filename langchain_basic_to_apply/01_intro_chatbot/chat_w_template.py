import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

chat = ChatOpenAI(temperature=0)

# >>> basic >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# res = chat.invoke([HumanMessage(content="Translate this sentence from English to French. I love programming.")])
# res = chat.invoke(messages)
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# >>> with system message >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# messages = [
#     SystemMessage(content="You are a helpful assistant that translates English to French."),
#     HumanMessage(content="Translate this sentence from English to French. I love people.")
# ]
# res = chat.invoke(messages)
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# >>> batch messages >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# batch_messages = [
#     [
#         SystemMessage(content="You are a helpful assistant that translates English to French."),
#         HumanMessage(content="Translate this sentence from English to French. I love programming.")
#     ],
#     [
#         SystemMessage(content="You are a helpful assistant that translates English to French."),
#         HumanMessage(content="Translate this sentence from English to French. I love artificial intelligence.")
#     ],
# ]

# res = chat.generate(batch_messages)

# print(res)
# print(res.llm_output['token_usage'])
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# >>> with template >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

sys_template = "You are a helpful assistant and cute that translates {input_language} to {output_language}."
human_template = "{human_text}"

system_message_prompt = SystemMessagePromptTemplate.from_template(sys_template)
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

print(type(chat_prompt))

#=== option 1
# res = chat.invoke(chat_prompt.format_prompt(input_language="English", output_language="Chinese Traditional", human_text="I love programming.").to_messages())

#=== option 2
# method 1
chain = LLMChain(llm=chat, prompt = chat_prompt)
# method 2
# chain = chat_prompt | chat

# method 1 and 2 will give different format of the result

# run will be deprecated
# res = chain.run(input_language="English", output_language="Chinese Traditional", human_text="I love programming.")

input = "I love apple."

res = chain.invoke({"input_language":"English", "output_language":"Chinese Traditional", "human_text":input})

#=== print
print(res)

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
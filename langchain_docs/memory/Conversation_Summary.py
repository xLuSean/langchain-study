import os
from dotenv import load_dotenv
load_dotenv()

from langchain.memory import ConversationSummaryMemory
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
# from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# memory = ConversationSummaryMemory(llm=OpenAI(temperature=0))
# memory.save_context({"input": "hi"}, {"output": "whats up"})

# res = memory.load_memory_variables({})
# print(res)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
## We can also get the history as a list of messages (this is useful if you are using this with a chat model).
# memory = ConversationSummaryMemory(llm=OpenAI(temperature=0), return_messages=True)
# memory.save_context({"input": "hi"}, {"output": "whats up"})

# res = memory.load_memory_variables({})
# print(res)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
## We can also utilize the predict_new_summary method directly.
# messages = memory.chat_memory.messages
# previous_summary = ""
# res = memory.predict_new_summary(messages, previous_summary)
# print(res)

########################################################
### Initializing with messages/existing summary
########################################################
history = ChatMessageHistory()
history.add_user_message("hi")
history.add_ai_message("hi there!")

# memory = ConversationSummaryMemory.from_messages(
#     llm=OpenAI(temperature=0),
#     chat_memory=history,
#     return_messages=True
# )

memory = ConversationSummaryMemory(
    llm=ChatOpenAI(temperature=0),
    # buffer="The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good because it will help humans reach their full potential.",
    chat_memory=history,
    return_messages=True
)

# res = memory.buffer
previous_summary = "The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good because it will help humans reach their full potential"

messages = memory.chat_memory.messages
print(messages)
res = memory.predict_new_summary(messages, previous_summary)
print(res)
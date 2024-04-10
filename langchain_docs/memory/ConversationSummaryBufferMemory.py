import os
from dotenv import load_dotenv
load_dotenv()
# =================================================================
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import OpenAI

llm = OpenAI()

#######################################################################
### Using memory with LLM
#######################################################################

#=== basic usage
# memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=10)
# memory.save_context({"input": "who is Smith's mother"}, {"output": "Mrs. Brown"})
# memory.save_context({"input": "Did Mrs.Brons has other children?"}, {"output": "Yes, she also has a daughter."})

# res = memory.load_memory_variables({})

# print(res)

#===
# 我们也可以将历史记录作为消息列表获取（这在使用聊天模型时很有用）。
# We can also get the history as a list of messages (this is useful if you are using this with a chat model).

# memory = ConversationSummaryBufferMemory(
#     llm=llm, max_token_limit=10, return_messages=True
# )
# memory.save_context({"input": "who is Smith's mother"}, {"output": "Mrs. Brown"})
# memory.save_context({"input": "Did Mrs.Brons has other children?"}, {"output": "Yes, she also has a daughter."})

# res = memory.load_memory_variables({})
# print(res.get("history"))


# We can also utilize the predict_new_summary method directly.
# 我们还可以直接利用predict_new_summary方法。

# messages = memory.chat_memory.messages
# previous_summary = ""
# res = memory.predict_new_summary(messages, previous_summary)

# print(res)

#######################################################################
### Using in a chain
#######################################################################
from langchain.chains import ConversationChain

conversation_with_summary = ConversationChain(
    llm=llm,
    # We set a very low max_token_limit for the purposes of testing.
    memory=ConversationSummaryBufferMemory(llm=OpenAI(), max_token_limit=40),
    verbose=True,
)
res = conversation_with_summary.predict(input="John and Ginger are in relation")
print(res)

# We can see here that there is a summary of the conversation and then some previous interactions
res = conversation_with_summary.predict(input="Who is Ginger's boyfriend?")
print(res)
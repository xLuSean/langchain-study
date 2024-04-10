from langchain.memory import ConversationBufferMemory

#=== basic
# memory = ConversationBufferMemory()
# memory.save_context({"input":"hi"}, {"output":"what's up"})
# print(memory.load_memory_variables({}))

#=== basic in chat mode
# memory = ConversationBufferMemory(return_messages=True)
# memory.save_context({"input":"hi"}, {"output":"what's up"})
# print(memory.load_memory_variables({}))

##########################################################
### Using in a chain
##########################################################
from langchain_openai import OpenAI
from langchain.chains import ConversationChain

llm = OpenAI(temperature=0)
conversation = ConversationChain(llm=llm, verbose=True, memory=ConversationBufferMemory())

res = conversation.predict(input="Mary and Jerry are married")
print(res)

res = conversation.predict(input="who is Mary's husband?")
print(res)
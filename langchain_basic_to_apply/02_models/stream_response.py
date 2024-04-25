from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAI, ChatOpenAI
from langchain_community.chat_models import ChatAnthropic
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import HumanMessage

# llm = OpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0)
# # llm = OpenAI(streaming=True, temperature=0)
# # llm = OpenAI(callbacks=[StreamingStdOutCallbackHandler()], temperature=0)
# resp = llm.invoke("Write me a song about quantum mechanics.")

chat = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0)
resp = chat.invoke([HumanMessage(content="Write me a song about sparkling water.")])

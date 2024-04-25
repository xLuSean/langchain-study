# not work as expected

import os
from dotenv import load_dotenv
load_dotenv()

import langchain
from langchain.cache import InMemoryCache
from langchain_openai import OpenAI, ChatOpenAI
from langchain.chains import LLMChain
import time

langchain.llm_cache = InMemoryCache()

# To make the caching really obvious, lets use a slower model.
# llm = OpenAI(model_name="gpt-3.5-turbo", n=2, best_of=2)
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

start_time = time.perf_counter()
res = llm.invoke("Tell me a joke")
middle_time = time.perf_counter()
print(res, f"Time taken: {middle_time - start_time:2f} seconds")

res = llm.invoke("Tell me another joke")
end_time = time.perf_counter()
print(res, f"Time taken: {end_time - middle_time:2f} seconds")
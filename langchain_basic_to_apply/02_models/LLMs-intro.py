from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAI

# ===================================================
# client = OpenAI()
# response = client.chat.completions.create(
#   model="gpt-3.5-turbo",
#   messages=[
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "Who won the world series in 2020?"},
#     {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
#     {"role": "user", "content": "Where was it played?"}
#   ]
# )
# print(response)

# ===================================================
llm = OpenAI(temperature=0.9)
res = llm.generate(["Hello, how are you?"])
print(res.generations,"\n##################################")
print(res.generations[0],"\n##################################")
print(res.generations[0][0].text,"\n##################################")
print(type(res.generations[0][0]))
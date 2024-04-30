from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="/Users/sean/Learn/AI/langchain/.env")

chat = ChatAnthropic(temperature=0, model_name="claude-3-haiku-20240307")
print(os.getenv("ANTHROPIC_API_KEY"))
print(os.getenv("OPENAI_API_KEY"))

# system = (
#     "You are a helpful assistant that translates {input_language} to {output_language}."
# )
# human = "{text}"
# prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

# chain = prompt | chat
# chain.invoke(
#     {
#         "input_language": "English",
#         "output_language": "Korean",
#         "text": "I love Python",
#     }
# )
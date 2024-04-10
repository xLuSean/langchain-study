##############################################
### description: using prompt template
##############################################

from dotenv import load_dotenv
load_dotenv()
import os
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
#===========================================================

# from getpass import getpass
# from langchain.chains import LLMChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

# OPENAI_API_KEY = getpass('OpenAI API Key:')
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

template = """Question: {question}
Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm = OpenAI()
# If you manually want to specify your OpenAI API key and/or organization ID, you can use the following:
# llm = OpenAI(openai_api_key="YOUR_API_KEY", openai_organization="YOUR_ORGANIZATION_ID")

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"

answer=llm_chain.invoke(question)
print(answer)

# print(llm('30字以內描述南極'))
from dotenv import load_dotenv
load_dotenv()
import os
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
# ================================================================================

from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain

llm = OpenAI(temperature=1)
template = """你是一個洗衣機製造商的工程人員，你會收到來自客服部門的問題，請提供客服部門關於這個問題的回答:
% 客服部門問題
{service_question}

你的回答:
"""
prompt_template = PromptTemplate(input_variables=["service_question"], template=template)
engeneer_chain = LLMChain(llm=llm, prompt=prompt_template)

template = """你是洗衣機製造商的客服人員，你會收到客戶關於機器的問題，你會將客戶的問題講述清楚之後，轉述給工程人員：
% 客戶問題
{user_question}

你對客戶問題的轉述:
"""
prompt_template = PromptTemplate(input_variables=["user_question"], template=template)
service_chain = LLMChain(llm=llm, prompt=prompt_template)

# 通过 SimpleSequentialChain 串联起来，第一个答案会被替换第二个中的問題，然后再进行询问
overall_chain = SimpleSequentialChain(chains=[service_chain, engeneer_chain], verbose=True)
review = overall_chain.invoke("我的洗衣機在運轉時產生異常振動和噪音")

# print(review)
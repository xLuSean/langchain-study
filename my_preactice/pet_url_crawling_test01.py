import os
from dotenv import load_dotenv
load_dotenv()
# ================================================================================

from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import LLMRequestsChain, LLMChain

# llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0)
llm = OpenAI( temperature=0)

template = """給出這個頁面關於動物知識的完整解答，去除標題、圖片、廣告、導航、網頁介紹等無關內容。我不需要知道這個網站的名稱，只需要知道這個網站的內容。 
>>> {requests_result} <<<"""

prompt = PromptTemplate(
    input_variables=["requests_result"],
    template=template
)

chain = LLMRequestsChain(llm_chain=LLMChain(llm=llm, prompt=prompt))
inputs = {
#   "url": "https://www.afurkid.com/Article/show/7832"
    "url": "https://sanimaldise.nvri.gov.tw/diseshow.aspx?cid=481F039F8C9F2CAF&docid=3A355E25F5256694"
}

response = chain.invoke(inputs)
print(response)
# Example taken from https://ithelp.ithome.com.tw/articles/10345264

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
import os

# 定義 Output Parser 函數，取出 html 程式的部分
def custom_parser(message) -> str:
	html_code = message.content.split("```")[1].replace("html", "", 1)
	return html_code

# 將 html 程式放進 flask 執行
def run_flask_app(html_code):
	# 如果不存在 templates 資料夾的話，會創建一個
	os.makedirs("templates", exist_ok=True)
	# 將解析完成的 html 程式寫入 index.html
	with open("templates/index.html", "w") as index_file:
		index_file.write(html_code)

	# 寫一個 flask 的程式
	app_code = f"""from flask import Flask, make_response, render_template
app = Flask(__name__)
@app.route('/')
def index():
	response = make_response(render_template("index.html"))
	return response
if __name__ == '__main__':
	app.run()
"""
	# 將上面 flask 程式寫入 app.py
	with open("app.py", "w") as file:
		file.write(app_code)
	# 執行 app.py
	os.system("python app.py")

# 選擇模型
llm = ChatOpenAI(model="gpt-4o")

# 指令
template = "請用 html 寫一個 {web} 的網頁給我，要包含{function}功能，包含js的部分也要幫我寫進去！"
prompt = PromptTemplate.from_template(template)

# 將所有 Runnable Chain 起來，並且傳入我要的網頁型態和功能
chain = prompt | llm | RunnableLambda(custom_parser) | RunnableLambda(run_flask_app)
chain.invoke({"web":"to-do list", "function":"新增修改刪除"})
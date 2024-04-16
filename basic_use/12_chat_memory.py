import os
from api_keys import OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# ================================================================================

from langchain.memory import ChatMessageHistory
from langchain_openai import ChatOpenAI


chat = ChatOpenAI(temperature=0)

# 初始化 MessageHistory 对象
history = ChatMessageHistory()

# 给 MessageHistory 对象添加对话内容
history.add_ai_message("你好！")
history.add_user_message("""有兩個人，稱作A和B。他們待在一個沒有窗戶的房間裡，房間裡面還有兩個空箱子1和2，和一顆球。
                         A把球放到1號箱子裡，然後離開房間。當A從外面回來時，他會去哪個箱子找球？""")

ai_response = chat(history.messages)
print(ai_response)

history.add_ai_message("""如果當A離開時，B把球從1號箱子拿出來移到2號箱子，當A回來的時候會去哪裡找球？""")

ai_response = chat(history.messages)
print(ai_response)
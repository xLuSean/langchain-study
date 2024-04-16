from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.llms.ollama import Ollama


chat001 = ChatMessageHistory()
chat001.add_user_message('My name is Amo.')

store = {'chat001': chat001,}


def get_chat_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


llm = Ollama(model='llama2')

prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a good assistant.'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('user', '{input}'),
])

chain = prompt | llm

with_message_history = RunnableWithMessageHistory(
    chain,
    get_chat_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

session_id = 'chat001'
input_text = input('>>> ')
while input_text.lower() != 'bye':
    if input_text:
        response = with_message_history.with_config(
            configurable={'session_id': session_id}
        ).invoke({'input': input_text})
        print(response)
    input_text = input('>>> ')
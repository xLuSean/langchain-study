# https://python.langchain.com/docs/modules/data_connection/text_embedding/

##############################################
### description: embedding test
##############################################
# import os
# from api_keys import OPENAI_API_KEY
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

from dotenv import load_dotenv
load_dotenv()
import os
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
#===========================================================
from langchain_openai import OpenAIEmbeddings
# from getpass import getpass

embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

embeddings = embeddings_model.embed_documents(
    [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!"
    ]
)
# print(len(embeddings),len(embeddings[0]))

embedded_query = embeddings_model.embed_query("What was the name mentioned in the conversation?")
print(embedded_query[:5])
# print(embedded_query)
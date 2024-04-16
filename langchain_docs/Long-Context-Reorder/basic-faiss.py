from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain_community.document_transformers.long_context_reorder import (
    LongContextReorder,
)
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores.chroma import Chroma
# TODO: 改成faiss
from langchain_openai import OpenAI

# Get embeddings.
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

texts = [
    "Basquetball is a great sport.",
    "Fly me to the moon is one of my favourite songs.",
    "The Celtics are my favourite team.",
    "This is a document about the Boston Celtics",
    "I simply love going to the movies",
    "The Boston Celtics won the game by 20 points",
    "This is just a random text.",
    "Elden Ring is one of the best games in the last 15 years.",
    "L. Kornet is one of the best Celtics players.",
    "Larry Bird was an iconic NBA player.",
]

# Create a retriever
retriever = Chroma.from_texts(texts, embedding=embeddings).as_retriever(
    search_kwargs={"k": 10}
)
query = "What can you tell me about the Celtics?"

# Get relevant documents ordered by relevance score
docs = retriever.get_relevant_documents(query)
print(f"=============== original: ==================\n{docs}")

# =============================================================================

# Reorder the documents:
# Less relevant document will be at the middle of the list and more relevant elements at beginning / end.
reordering = LongContextReorder()
reordered_docs = reordering.transform_documents(docs)

# Confirm that the 4 relevant documents are at beginning and end.
print(f"=============== reordered: ==================\n{reordered_docs}")

# =============================================================================
# We prepare and run a custom Stuff chain with reordered docs as context.

# Override prompts
# for StuffDocumentsChain
document_prompt = PromptTemplate(
    input_variables=["page_content"], template="{page_content}") #因為document的內容是"page_content"

# document_variable_name = "context"

llm = OpenAI()

stuff_prompt_override = """Given this text extracts:
-----
{context}
-----
Please answer the following question:
{query}"""

# for LLMChain
prompt = PromptTemplate(
    template=stuff_prompt_override, input_variables=["context", "query"]
)

# Instantiate the chain
llm_chain = LLMChain(llm=llm, prompt=prompt)

chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_prompt=document_prompt,
    document_variable_name="context",
    # document_variable_name=document_variable_name,
)
res = chain.run(input_documents=reordered_docs, query=query)
print(res)
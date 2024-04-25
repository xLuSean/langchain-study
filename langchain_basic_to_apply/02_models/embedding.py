from langchain_community.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings


embeddings = HuggingFaceEmbeddings(model_name="bert-base-uncased")
text = "This is a test document."
query_result = embeddings.embed_query(text)
doc_result = embeddings.embed_documents([text])


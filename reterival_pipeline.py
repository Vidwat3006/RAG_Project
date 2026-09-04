import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chroma/db"
embeddingModel = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddingModel,
    collection_metadata={"hnsw:space":"cosine"}
)
query = "What island does SpaceX lease for its launch in the Pacific?"

retriever = db.as_retriever(search_kwargs={"k":5})
relevant_docs = retriever.invoke(query)
print(f"User Query: {query}")
# print("-----Context-----")
# for i,doc in enumerate(relevant_docs,1):
#     print(f"Document {i}:\n{doc.page_content}\n")


combined_input = f"""
Answer the following query using ONLY the information provided in the documents.

Query:
{query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Instructions:
- Use ONLY the information from the documents above.
- Do NOT use your own knowledge.
- Do NOT search the internet.
- If the answer cannot be found in the documents, say:
  "The answer for this query is not available in the given documents."
- Do not make assumptions or infer information that is not supported by the documents.
"""

model = ChatGroq(model="openai/gpt-oss-120b",api_key=os.getenv("GROQ_API_KEY"))
messages=[
    SystemMessage(content="You are a helpful RAG assistant. Answer only from the provided context."),
    HumanMessage(content = combined_input)
]
result = model.invoke(messages)
print("\n----Generated Response----")
print(result.content)
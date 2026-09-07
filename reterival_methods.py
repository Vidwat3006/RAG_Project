from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "./db/chroma/db"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
)

query = "how much did Microsoft pay to acquire GitHub?"

# print("===Similarity Search===")
# retervier = db.as_retriever(search_kwargs={"k":3})
# docs = retervier.invoke(query)

# print(f"Reterived {len(docs)}")
# for i,doc in enumerate(docs,1):
#     print(f"Document {i}")
#     print(doc.page_content)
# print("-"*60)

# print("\n===Method 2: Similarity with Score Threshold===")
# retervier = db.as_retriever(
#     search_type = "similarity_score_threshold",
#     search_kwargs = {
#         "k":3,
#         "score_threshold":0.3
#     }
# )

# docs = retervier.invoke((query))

# for i,doc in enumerate(docs,1):
#     print(f"Document {i}")
#     print(doc.page_content)


#Gives different aspects(dicersity = different chunks from the ones already selected) with similiar relevance
print("====Method 3:Maximum Marginal Relevance====")
reteriver = db.as_retriever(
    search_type = "mmr",
    search_kwargs={
        "k":3, #final number of chunks
        "fetch_k":10, #initial pool of chunks
        "lambda_mult":0.5 #0=Max diversity 1=Max relevance
    }
)

docs = reteriver.invoke(query)
for i,doc in enumerate(docs,1):
    print(f"Document {i}")
    print(doc.page_content)
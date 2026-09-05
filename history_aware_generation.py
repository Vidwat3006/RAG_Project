import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chroma/db"
embeddingModel = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddingModel,
    collection_metadata={"hnsw:space":"cosine"}
)

chat_history = []
model = ChatGroq(model="openai/gpt-oss-120b",api_key=os.getenv("GROQ_API_KEY"))

def ask_question(user_question):

    print(f"You asked: {user_question}")

    if chat_history:
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]

        result = model.invoke(messages)
        search_question = result.content

    else:
        search_question = user_question


    retriever = db.as_retriever(search_kwargs={"k":3})
    docs = retriever.invoke(search_question)
    print(f"User Query: {search_question}")

    combined_input = f"""Based on the following documents, please answer this question: {user_question}

    Documents:
    {"\n".join([f"- {doc.page_content}" for doc in docs])}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
    """

    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history."),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]
    
    result = model.invoke(messages)
    answer = result.content
    
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))
    
    print(f"Answer: {answer}")
    return answer


def start_chat():
    print("Search Anything. Type quit to exit")
    while True:
        question = input("\nYour question: ")
        if question.lower()=="quit":
            print("GoodBye!!")
            break
        ask_question(question)

if __name__ == "__main__":
    start_chat()
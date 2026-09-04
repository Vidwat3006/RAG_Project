import os
from langchain_community.document_loaders import TextLoader,DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path):
    print(f"Loading all the documents from {docs_path}:")
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist.")
    loader = DirectoryLoader(
        path = docs_path,
        glob= "*.txt",
        loader_kwargs={"encoding": "utf-8"},
        loader_cls=TextLoader
    )
    documents = loader.load()
    if(len(documents)) == 0:
        raise FileNotFoundError(f"No .txt files found in the directory {docs_path}")
    print(len(documents))
    return documents

def split_documents(documents,chunk_size=800,chunk_overlap=0):
    text_splitter = CharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    print(len(chunks))
    print(type(chunks))
    return chunks


def vectorize_and_store(chunks,persist_directory="db/chroma/db"):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("------Vector Store-------")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding= embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space":"cosine"}
    )
    print("----Completed DB Creation ----")
    return vectorstore


def main():
    print("Main function")


    #Load all the required documents

    documents = load_documents(docs_path="docs")

    #Chunk the documents

    splitted_docs = split_documents(documents)

    #Embbed the data and store in vectorDB(Chroma)

    vectorstore = vectorize_and_store(splitted_docs)


if __name__ == "__main__":
    main()
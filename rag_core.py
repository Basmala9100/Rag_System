# rag_core.py
import os
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import uuid

load_dotenv()

# Initialize embeddings
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite-preview-06-17",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

def embed_and_store(texts, persist_directory="models/chroma_db"):
    unique_dir = f"models/chroma_db_{uuid.uuid4()}"
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=model,
        persist_directory=unique_dir
    )
    # Do NOT call .persist() explicitly in langchain_chroma—it saves automatically
    return vectorstore

def rag_query(query, vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":5})
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"Answer the question based on these documents:\n{context}\nQuestion: {query}\nAnswer:"

    answer = llm.invoke(prompt).content
    return answer, docs

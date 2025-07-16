# api_rag.py
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import os
import uuid

from preprocessing import preprocess_data, make_chunks
from rag_core import embed_and_store, rag_query

app = FastAPI(title="Universal RAG API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use in-memory vectorstore (no persistence to avoid old data mix)
vectorstore = None

class QueryInput(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "Welcome to the Universal RAG API!"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global vectorstore

    file_path = f"uploaded_{uuid.uuid4()}_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Read file
    if file.filename.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file.filename.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        return {"error": "Unsupported file format. Please upload CSV or Excel."}

    # Preprocess and embed
    text_data = preprocess_data(df)
    chunks = make_chunks(text_data)
    vectorstore = embed_and_store(chunks)  # In-memory store

    os.remove(file_path)

    return {"message": f"{file.filename} uploaded and indexed successfully."}

@app.post("/query")
def query_endpoint(input_query: QueryInput):
    global vectorstore
    if vectorstore is None:
        return {"error": "No file uploaded yet."}

    answer, docs = rag_query(input_query.question, vectorstore)

    return {
        "question": input_query.question,
        "answer": answer,
        "retrieved_docs": [doc.page_content for doc in docs]
    }

@app.get("/ask")
def ask(question: str = Query(...)):
    global vectorstore
    if vectorstore is None:
        return {"error": "No file uploaded yet."}

    answer, docs = rag_query(question, vectorstore)

    return {
        "question": question,
        "answer": answer,
        "retrieved_docs": [doc.page_content for doc in docs]
    }

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from for_web import index_excel_file, rag_pipeline, query_rag
from fastapi import Query


app = FastAPI(title="RAG System", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class QueryInput(BaseModel):
    """Model for input query."""
    question: str
    
@app.get("/")
def read_root():
    return {"message": "Welcome to the RAG System API!"}

@app.post("/query")
def rag_endpoint(input_query: QueryInput):
    answer, documents, metadatas = rag_pipeline(input_query.question)
    return {
        "question": input_query.question,
        "answer": answer,
        "documents": documents,
        "metadatas": metadatas  
    }
    
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    
    with open(file.filename, "wb") as f:
        f.write(content)
        
    index_excel_file(file.filename)
    # Here you can add logic to process the uploaded file if needed
    return {"filename": f"{file.filename} uploaded and indexed successfully"}


@app.get("/ask")
async def ask_question(question: str = Query(...)):
    try: 
        answer = query_rag(question)
        return {"question": question, "answer": answer}
    except Exception as e:
        return {"error": str(e)}
    
    
@app.post("/query")
def query_rag_endpoint(request: QueryInput):
    result = rag_pipeline(request.question)
    return result
    
class QueryResponse(BaseModel):
    """Model for query response."""
    question: str
    answer: str
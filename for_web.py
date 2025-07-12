import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers  import SentenceTransformer 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
import chromadb 
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite-preview-06-17",
    google_api_key= os.getenv("GOOGLE_API_KEY")
)

model = SentenceTransformer("all-MiniLM-L6-v2")

Chroma_client = chromadb.Client()
collection = Chroma_client.get_or_create_collection(name="student_reviews")

vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=model
    )    
retriever = vectorstore.as_retriever(search_type= "similarity", search_kwargs={"k": 5})

def get_embedding(text):
    return model.encode(text)

#query Chroma for Similar Chunks
def search_and_retrieve(query, top_k=5):
    query_embedding = get_embedding(query)

    reterival_info = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas"]
    )
    documents = reterival_info["documents"][0]
    metadatas = reterival_info["metadatas"][0]

    return documents, metadatas


def generate_prompt(query, documents):
    context = "\n".join(documents)
    prompt = f"""Answer the question based on students' reviews
              Some of the feedback recieved:
              {context}
              question: {query}
              answer:"""
    return prompt


def rag_pipeline(query, k=5):
    # Retrieve relevant documents from the vector database
    retrieved_documents, metadatas = search_and_retrieve(query, top_k=k)

    # Generate the prompt for the LLM using the retrieved documents
    prompt = generate_prompt(query, retrieved_documents)

    # Generate the answer using the language model
    answer = llm.invoke(prompt).content

    return answer, retrieved_documents, metadatas


def index_excel_file(file_path):
    # Function to index an Excel file into the vector database
    df = pd.read_excel(file_path)
    #list_of_texts = df.apply(lambda row: " ".join([str(cell) for cell in row]), axis=1).tolist()
    #print(list_of_texts[:5])
    
    required_columns = ['Course Name', 'Rating', 'Comment']
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' is missing from the Excel file.")
        
    df = df.dropna(subset=['Comment'])
    
    documents = []
    for _, row in df.iterrows():
        content = f"Course Name: {row['Course Name']}\nRating: {row['Rating']}\nComment: {row['Comment']}"
        metadata = {
            "course_name": row['Course Name'],
            "rating": row['Rating'],
            "Student Name": row.get('Student Name', None),
            "timestamp": row.get('Timestamp', None)
        }
        documents.append(Document(page_content=content, metadata={"source": metadata}))
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_documents = text_splitter.split_documents(documents)
    filtered_documents = filter_complex_metadata(split_documents)
    vectorstore = Chroma.from_documents(
        documents=filtered_documents,
        embedding=model,
        persist_directory="chroma_db"
    )
    vectorstore.persist()
    
    print(f"Indexed {len(split_documents)} documents from {file_path} into the vector database.")
    
    
def query_rag(question: str):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    result = qa_chain.invoke({"query": question})
    return result["result"]
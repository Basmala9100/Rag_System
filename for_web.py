model = SentenceTransformer('all-MiniLM-L6-v2')

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
    answer = llm.invoke(prompt)

    return answer, retrieved_documents, metadatas
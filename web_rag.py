import streamlit as st
from for_web import rag_pipeline


st.set_page_config(page_title="RAG Q&A", layout="centered")
st.title("RAG Chat")
        
def main():
    st.write("Welcome to the RAG Q&A system!")
    st.text_input("Enter your question here and click 'Submit' to get the answer:", key="query_input")
    if st.button("Submit"):
        query = st.session_state.query_input
        if query:
            answer, documents, metadatas = rag_pipeline(query)
            st.write("Answer:", answer)
            
        else:
            st.warning("Please enter a question.")


if __name__ == "__main__":
    main()
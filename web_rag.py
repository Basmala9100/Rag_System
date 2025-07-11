import streamlit as st
from for_web import rag_pipeline


st.set_page_config(page_title="RAG Q&A", layout="centered")
st.title("RAG Chat")
query = st.text_input("Ask something:", placeholder="e.g. What is RAG?")

        
def main():
    st.write("Welcome to the RAG Q&A system!")
    st.write("Enter your question in the input box below and click 'Submit' to get an answer.")
    st.text_input("Enter your question here:", key="query_input")
    if st.button("Submit"):
        query = st.session_state.query_input
        if query:
            answer, documents, metadatas = rag_pipeline(query)
            st.write("Answer:", answer)
            st.write("Retrieved Documents:")
            for doc in documents:
                st.write(doc)
            st.write("Metadata:")
            for meta in metadatas:
                st.write(meta)
        else:
            st.warning("Please enter a question.")


if __name__ == "__main__":
    main()
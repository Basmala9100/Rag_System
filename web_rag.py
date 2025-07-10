
import streamlit as st
import requests

# You can import your backend logic here or define the Gemini function inline
def ask_gemini(query):
    return f"Response from Gemini: {query[::-1]}"

st.set_page_config(page_title="Gemini Q&A", layout="centered")
st.title("Gemini Chat")

query = st.text_input("Ask something:", placeholder="e.g. What is RAG?")

if st.button("Submit") and query:
    with st.spinner("Thinking..."):
        response = ask_gemini(query)
        st.success(response)

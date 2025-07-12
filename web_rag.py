import streamlit as st
import psutil
import time     
from for_web import rag_pipeline

if 'query_count' not in st.session_state:
    st.session_state.query_count = 0


def get_system_info():
    st.sidebar.title("System Information: ")
    
    cpu_usage = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    memory_usage = memory.percent
    disk = psutil.disk_usage('/')
    
    st.sidebar.markdown(f"CPU Usage: {cpu_usage}%")
    st.sidebar.markdown(f"Memory Usage: {memory_usage}%")
    st.sidebar.markdown(f"Disk Usage%: {disk.percent}%")
    st.sidebar.markdown(f"Avaliable Memory: {round(memory.available / (1024 ** 3), 2)} GB")
    st.sidebar.markdown(f"Used Memory: {round(memory.used / (1024 ** 3), 2)} GB")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"Query Count: {st.session_state.query_count}")
    
        
def main():
    st.set_page_config(page_title="RAG Q&A", layout="centered")
    st.title("RAG Chat")
    st.write("Welcome to the RAG Q&A system!")
    
    with st.form("query_form"):
        user_question = st.text_input("Enter your question:", key="query_input")
        submit = st.form_submit_button("Ask")
        
        if submit and user_question:
            with st.spinner("Processing your question..."):
                answer, documents, metadatas = rag_pipeline(user_question)
                st.session_state.query_count += 1
                
                st.write("Answer:", answer)
        elif submit:
            st.warning("Please enter a question before submitting.")


if __name__ == "__main__":
    get_system_info()
    main()
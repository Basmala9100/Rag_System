import streamlit as st
import psutil
import time     
from for_web import rag_pipeline

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
    get_system_info()
    main()
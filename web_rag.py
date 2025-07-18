# web_rag.py
import streamlit as st
import pandas as pd
import psutil
import platform
import os

#<<<<<<< HEAD
from preprocessing import preprocess_data, make_chunks
from rag_core import embed_and_store, rag_query

def system_info():
    st.sidebar.markdown("## System Monitoring")
    st.sidebar.write(f"CPU Usage: {psutil.cpu_percent()}%")
    st.sidebar.write(f"RAM Usage: {psutil.virtual_memory().percent}%")

    # Dynamically get the correct drive for Windows/Linux/Mac
    if os.name == 'nt':  # Windows
        current_drive = os.path.splitdrive(os.getcwd())[0] + '\\'
    else:  # Linux/Mac
        current_drive = '/'

    try:
        disk_percent = psutil.disk_usage(current_drive).percent
        st.sidebar.write(f"Disk Usage ({current_drive}): {disk_percent}%")
    except Exception as e:
        st.sidebar.write(f"Disk Info: {str(e)}")

    st.sidebar.write(f"Platform: {platform.system()} {platform.release()}")

def main():
    st.title("📊 RAG System - Streamlit Interface")
    st.write("Upload your Excel/CSV file and ask questions!")
#=======
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
#>>>>>>> e9676e9411fd01e701bd20815c2534deee73e802

    system_info()  # Show system info in sidebar

    uploaded_file = st.file_uploader("Upload your file", type=["csv", "xlsx"])

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("Data Preview")
        st.dataframe(df.head())

        texts = preprocess_data(df)
        chunks = make_chunks(texts)

        st.info("Embedding and storing data... Please wait.")
        vectorstore = embed_and_store(chunks)

        st.success("Data indexed successfully! You can now ask questions.")

        query = st.text_input("Ask your question:")

        if query:
            answer, docs = rag_query(query, vectorstore)
            st.subheader("Answer")
            st.write(answer)

            with st.expander("Show Retrieved Documents"):
                for doc in docs:
                    st.write(doc.page_content)

if __name__ == "__main__":
    main()

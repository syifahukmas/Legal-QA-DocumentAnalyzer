import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_openai import ChatOpenAI
import os
# from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.document_loaders import PyPDFLoader
from typesense import Client
import json
from langchain_core.prompts import ChatPromptTemplate
import numpy as np
from scipy.spatial.distance import cosine
from utils import chunk_document, create_superheader_context, process_text_in_chunks, hybrid_search, create_llm, create_chain

# # Load environment variables
# load_dotenv()

upload_directory = "uploads"

def main():
    st.title("LangChain Streamlit Interface with Typesense")

    # Sidebar options
    st.sidebar.header("Configuration")
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    typesense_api_key = st.sidebar.text_input("Typesense API Key", type="password")
    typesense_host = st.sidebar.text_input("Typesense Host")

    uploaded_file = None # tambahan 1/7/2025
    
    if api_key:
        try:
            llm = create_llm(api_key)
            st.success("✅ API Key berhasil digunakan!")
        except ValueError as e:
            st.error(f"⚠️ Error: {e}")
    else:
        st.warning("⚠️ Masukkan OpenAI API Key untuk melanjutkan.")

    # Initialize Typesense client
    if typesense_api_key and typesense_host:
        typesense_client = Client({
            'nodes': [{
                'host': typesense_host,
                'port': '8108',
                'protocol': 'http'
            }],
            'api_key': typesense_api_key,
            'connection_timeout_seconds': 2
        })
    else:
        typesense_client = None

    # Select operation
    operation = st.sidebar.selectbox(
        "Choose Operation", ["Upload and Process PDF", "Question-Answer Retrieval"]
    )

    if operation == "Upload and Process PDF":
        st.subheader("Upload and Process PDF", divider="grey")
        if not os.path.exists(upload_directory):
            os.makedirs(upload_directory)

        file_upload = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

        if file_upload is not None:
            for uploaded_file in file_upload:
                # Simpan file ke direktori tujuan
                file_path = os.path.join(upload_directory, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                # st.write(f"File {uploaded_file.name} has been saved to {file_path}.")
            
            if uploaded_file: # tambahan 1/7/2025 : untuk menghilangkan error pada streamlit
                file_process = os.path.join(upload_directory, uploaded_file.name)
                loader = UnstructuredLoader(
                    file_process, chunking_strategy="basic", split_pdf_page=True, max_characters=1000000
                    )
                documents = loader.load()
                # st.write("documents loaded successfully!")
                # Chunk knowledge
                # Raw text dari dokumen
                raw_text = documents[0].page_content

                # Konversi raw_text menjadi daftar baris
                doc_lines = raw_text.splitlines()

                # Panggil fungsi chunk_document
                chunks = chunk_document(doc_lines,uploaded_file.name)
            
                # Asumsikan 'chunks' adalah hasil dari chunk_document(raw_text)
                chunks_json = json.dumps(chunks, indent=4, ensure_ascii=False)
                print(json.dumps(chunks, indent=4, ensure_ascii=False))
                # st.write("Chunks:", chunks_json)

                superheader = create_superheader_context(raw_text, api_key, raw_text)
                print("Superheader:", superheader)
                st.write(superheader)
            
                chunks_list = json.loads(chunks_json)
                # Proses teks dalam chunks
                all_summaries = process_text_in_chunks(typesense_client, chunks_list, result=superheader)
                # st.write("#All Summaries: ", all_summaries)

            else: # tambahan 1/7/2025
                st.write("No file was processed. Please upload a file.") # tambahan 1/7/2025

        # else: # tambahan 1/7/2025
        #     st.write("Please upload a PDF file.") # tambahan 1/7/2025

    elif operation == "Question-Answer Retrieval":
        st.subheader("Question-Answer Retrieval", divider="grey")
        collection_name = st.text_input("Enter Collection Name for Retrieval")
        query = st.text_input("Enter your question:")

        if st.button("Submit Query"):
            if not api_key:
                st.error("Please provide OpenAI API Key in the sidebar.")
            elif not typesense_client or not collection_name:
                st.error("Please configure Typesense in the sidebar and provide collection name.")
            else:
                # Retrieve matching chunks
                try:
                    results = hybrid_search(query, typesense_client, collection_name)
                    print("Hasil pencarian:")
                    
                    if results:
                        for i, (score, text) in enumerate(results, start=1):
                            print(f"{i}. (Score: {score}) {text}")
                            # st.write(f"{i}. (Score: {score}) {text}")
                        context = results
                        # st.write("Context:", context)
                        # Perform QA with context
                        embeddings = OpenAIEmbeddings()
                        prompt = f"""You are an AI assistant. Use the following context to answer the question:
                                    Context: {results}
                                    Question: {query}
                                    Answer:"""

                        response = ChatOpenAI(
                                    api_key=api_key,
                                    model="gpt-4o-mini",
                                    temperature=0,
                                    max_retries=2,
                                    timeout=120,
                                )
                        st.subheader("Response", divider="grey")
                        st.write(response.invoke(prompt).content)
                    else:
                        st.warning("No relevant chunks found.")
                except Exception as e:
                    st.error(f"Error during retrieval: {e}")

if __name__ == "__main__":
    main()

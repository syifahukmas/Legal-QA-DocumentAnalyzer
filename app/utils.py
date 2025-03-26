import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
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
import re

SYSTEM_CREATE_CHUNK_CONTEXT = """
    You are an advanced Retrieval-Augmented Generation assistant tasked with processing and organizing legal documents into structured and coherent chunks.

    ### Instructions:
    1. **You will be provided with**:
    - **Superheader:** The overarching context and theme of the document.
    - **Previous Context:** The context summary generated from preceding chunks.
    - **Current Chunk:** A new excerpt from the document represented as a JSON element, containing hierarchical and descriptive information. The chunk may include:
            - Title of the document (judul_dokumen)
            - Regulation number (nomor_peraturan)
            - Chapter information (bab)
            - Chapter title (judul_bab)
            - Section title (judul_bagian)
            - Article number (pasal)
            - Content (konten)

    2. **Hierarchy Extraction**:
    - Identify the structure of the document based on the JSON keys:
            - judul_dokumen: Overall title of the document.
            - nomor_peraturan: Regulation number or identifier.
            - bab: Chapter number.
            - judul_bab: Title of the chapter.
            - judul_bagian: Title of the section within a chapter.
            - pasal: Article number or identifier.
        - Clearly delineate the boundaries of each hierarchy.

    3. **Content Chunking**:
    - For each identified section (BAB or Pasal), extract its content using the chunk structure.
    - Include relevant context such as the chapter, section, or previous content for continuity.
    - Include all non-null fields in the summary:
            - If a field is null, exclude it from the summary.
            - Use a logical, human-readable format to present the information in the output.

    4. **Contextual Analysis**:
    - Combine relevant details from the superheader, the hierarchy, and the content of the current chunk.
    - Maintain logical flow between chunks by leveraging the previous context and overarching themes.
    - Analyze the **current chunk** to extract its main ideas and integrate them into a concise, coherent context.
    - Ensure that the generated context reflects the document's overall theme and maintains logical flow.
    - Avoid introducing information outside the provided content.
    - Answer in markdown format and in Bahasa Indonesia.

    5. **Output Format**:
    - Use the following markdown format to present each chunk:

    ### Hubungan dengan Konteks Sebelumnya:
        <Insert the concise context for the current chunk here. Focus on identifying and describing the relationship between the current article (pasal) and the previous context (bab, bagian, or document-level theme). Indicate whether the current chunk belongs to the same chapter, section, or regulation as the previous context, and highlight any transitions or connections. Avoid repeating detailed content summaries.>

    ### Ringkasan Konten:
    <Content Summary in paragraph form. Prioritize mentioning the article first, must be followed by information such as judul_dokumen, nomor_peraturan, bab, judul_bagian, and finally the explanation of the article>

    Answer strictly in the format provided.
    """

USER_CREATE_CHUNK_CONTEXT = """
    Saya akan memberikan superheader, konteks sebelumnya, dan potongan text dari sebuah dokumen berbahasa Indonesia.
    Berikut adalah informasi yang saya berikan:

    -- Superheader --
    {superheader}

    -- Konteks Sebelumnya --
    {previous_context}

    -- Text saat ini --
    {current_chunk}
    """

API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
        api_key=API_KEY,
        model="gpt-4o-mini",
        temperature=0,
        max_retries=2,
        timeout=120,
    )

prompt = ChatPromptTemplate(
        [
            ("system", SYSTEM_CREATE_CHUNK_CONTEXT),
            ("user", USER_CREATE_CHUNK_CONTEXT),
        ]
    )

chain = prompt | llm

def chunk_document(doc,judul_dokumen):
    chunks = []
    current_bab = None
    current_bagian = None
    current_pasal = None
    current_chunk = []
    judul_dokumen = judul_dokumen
    nomor_peraturan = None

    total_lines = len(doc)  # Total jumlah baris dalam dokumen
    progress_text = "Processing document. Please wait..."
    progress_bar = st.progress(0, text=progress_text)

    for idx, line in enumerate(doc):
        line = line.strip()

        # Ubah ke huruf kecil untuk perbandingan tanpa sensitivitas
        line_lower = line.lower()

        # Deteksi Judul Dokumen
        # if "peraturan presiden republik indonesia" in line_lower and not judul_dokumen:
        #     judul_dokumen = line
        #     continue

        # Deteksi Nomor Peraturan
        # Benerin lagi disini
        if re.match(r"^\s*nomor\b", line_lower) and not nomor_peraturan:
            nomor_peraturan = line
            continue

        # Deteksi BAB
        if line_lower.startswith("bab"):
            if current_chunk:
                chunks.append({
                    "bab": current_bab,
                    "judul_bagian": current_bagian,
                    "pasal": current_pasal,
                    "konten": " ".join(current_chunk),
                })
            current_bab = line
            current_bagian = None
            current_pasal = None
            current_chunk = []

        # Deteksi Bagian seperti "Menimbang", "Mengingat", "Menetapkan", dll.
        elif line_lower.startswith("bagian") or line_lower in ["menimbang", "mengingat", "menetapkan", "memperhatikan"]:
            if current_chunk:
                chunks.append({
                    "bab": current_bab,
                    "judul_bagian": current_bagian,
                    "pasal": current_pasal,
                    "konten": " ".join(current_chunk),
                })
            current_bagian = line
            current_pasal = None
            current_chunk = []

        # Deteksi Pasal
        elif line_lower.startswith("pasal"):
            if current_chunk:
                chunks.append({
                    "bab": current_bab,
                    "judul_bagian": current_bagian,
                    "pasal": current_pasal,
                    "konten": " ".join(current_chunk),
                })
            current_pasal = line
            current_chunk = []

        # Tambahkan baris lainnya ke dalam chunk
        else:
            current_chunk.append(line)

        # Perbarui progress bar
        percentage = int(((idx + 1) / total_lines) * 100)
        progress_bar.progress(percentage, text=f"{progress_text} ({percentage}%)")

    # Tambahkan chunk terakhir
    if current_chunk:
        chunks.append({
            "bab": current_bab,
            "judul_bagian": current_bagian,
            "pasal": current_pasal,
            "konten": " ".join(current_chunk),
        })

    # Tambahkan informasi judul dan nomor peraturan di awal
    if judul_dokumen or nomor_peraturan:
        chunks.insert(0, {
            "judul_dokumen": judul_dokumen,
            "nomor_peraturan": nomor_peraturan,
            "konten": None,
        })

    # Selesaikan progress bar
    progress_bar.progress(100, text="Documents loaded successfully")

    return chunks

def create_superheader_context(text, api_key, raw_text):
    chunk_superheader = raw_text[:5000]
    SYSTEM_CREATE_SUPERHEADER_CONTEXT = """
    You are an advanced Retrieval-Augmented Generation assistant for Legal Understanding.
    Your role is to process chunks of a document while maintaining context coherence and relevance.
    You will be given excerpt from a document beginnings in Indonesian Language.

    Always ensure your responses:
    - Reflect the overarching theme and context of the document.
    - Incorporate critical details while avoiding redundancy.
    - Avoid assumptions outside the provided context unless explicitly directed.
    - Your answer will be a basis to generate concise and coherent summaries, answer specific questions, or synthesize insights.
    - You have to answer in Markdown formattings and in Bahasa Indonesia.

    This will be your Markdown Structures:

    # Judul: <content of the title include "Nomor" for regulation number>
    # Latar Belakang: <background content>
    # Tujuan: <goals content>
    # Ringkasan: <summaries of content>

    """
    # Digunakan untuk memberikan konteks kepada model mengenai apa yang akan dilakukan oleh model dari sudut pandang user
    USER_CREATE_SUPERHEADER_CONTEXT = """
    Saya akan memberikan potongan awal dari sebuah dokumen berbahasa Indonesia.
    Berikut adalah text awal dari dokumen tersebut

    {text}
    """

    load_dotenv()
    API_KEY = os.getenv("OPENAI_API_KEY")

    llm = ChatOpenAI(
        api_key=API_KEY,
        model="gpt-4o-mini",
        temperature=0,
        max_retries=2,
        timeout=120,
    )

    prompt = ChatPromptTemplate(
        [
            ("system", SYSTEM_CREATE_SUPERHEADER_CONTEXT),
            ("user", USER_CREATE_SUPERHEADER_CONTEXT),
        ]
    )

    chain = prompt | llm

    result = chain.invoke(input={"text": chunk_superheader}).content
    return result

# Fungsi untuk menambahkan chunk dengan embedding
def add_chunk_with_embedding(client, chunk_id, superheader, previous_context):
    try:
        # Generate embedding dari previous_context
        previous_context_embedding = get_embedding(previous_context)
        previous_context_embedding_str = [str(val) for val in previous_context_embedding]  # Ubah ke string array

        document = {
            "id": chunk_id,
            "superheader": superheader,
            "previous_context": previous_context,
            "previous_context_embedding": previous_context_embedding_str  # Simpan embedding previous_context
        }

        # Tambahkan dokumen ke Typesense
        client.collections["chunks_coba"].documents.create(document)
        print(f"Chunk {chunk_id} berhasil ditambahkan dengan embedding!")
    except Exception as e:
        print(f"Error menambahkan chunk {chunk_id}: {e}")

def process_text_in_chunks(client, chunks_list, result):
    all_summaries = []
    previous_context = "-"  # Konteks awal
    end_pos = len(chunks_list)  # Panjang teks

    # Inisialisasi Streamlit progress bar
    progress_text = "Processing chunks. Please wait..."
    progress_bar = st.progress(0, text=progress_text)

    for i in range(end_pos):
        current_chunk = chunks_list[i]  # Ambil chunk saat ini

        # Invoke LLM untuk memproses chunk saat ini
        try:
            chunk_context = chain.invoke(
                input={
                    "superheader": result,
                    "previous_context": previous_context,
                    "current_chunk": current_chunk,
                }
            ).content
        except Exception as e:
            print(f"Error memproses chunk pada posisi {i}: {e}")
            continue

        # Membuat ID unik untuk chunk
        judul = chunks_list[0].get("nomor_peraturan", "")
        chunk_id = f"{judul}_{i+1}"

        # Tambahkan chunk dengan embedding ke Typesense
        add_chunk_with_embedding(client, chunk_id, result, previous_context)

        # Simpan rangkuman dan perbarui konteks sebelumnya
        all_summaries.append(chunk_context)
        previous_context = chunk_context

        # Perbarui progress bar
        percentage = int(((i + 1) / end_pos) * 100)
        progress_bar.progress(percentage, text=f"{progress_text} ({percentage}%)")

    # Selesaikan progress bar
    progress_bar.progress(100, text="Document processed into chunks successfully!")

    return all_summaries

# Fungsi untuk mendapatkan embedding dari teks
def get_embedding(text):
    embedding_model = OpenAIEmbeddings()
    try:
        embedding = embedding_model.embed_query(text)  # Menghasilkan list numerik
        return embedding
    except Exception as e:
        print(f"Error mendapatkan embedding: {e}")
        return []

def search_by_semantic_embedding(query, client, collection_name, threshold=0.1):
    # Generate the query embedding
    query_embedding = get_embedding(query)
    if query_embedding is None:
        return []

    # Normalize the query embedding
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    # Fetch documents from the specified collection in the client
    results = client.collections[collection_name].documents.export()
    documents = [json.loads(doc) for doc in results.splitlines()]
    scored_docs = []

    for doc in documents:
        # Convert the stored context embedding to a numpy array
        try:
            embedding = np.array([float(val) for val in doc['previous_context_embedding']])
            embedding = embedding / np.linalg.norm(embedding)  # Normalize the document embedding
        except (ValueError, KeyError):
            continue  # Handle missing or invalid embeddings

        # Compute the cosine similarity using scipy
        similarity = 1 - cosine(query_embedding, embedding)  # 1 - cos_distance gives us similarity

        if similarity > threshold:  # Only consider documents above threshold
            scored_docs.append((similarity, doc['previous_context']))

    # Sort documents by score in descending order
    scored_docs.sort(reverse=True, key=lambda x: x[0])
    
    # Return the top 5 results
    return scored_docs[:5]


def keyword_search(query, client, collection_name):
    # A simple keyword search that checks if the query is in the document
    results = client.collections[collection_name].documents.export()
    documents = [json.loads(doc) for doc in results.splitlines()]

    matched_docs = []
    for doc in documents:
        if query.lower() in doc['previous_context'].lower():
            matched_docs.append((1.0, doc['previous_context']))  # Use a high score for keyword matches

    return matched_docs

def hybrid_search(query, client, collection_name, threshold=0.1):
    # Perform keyword search
    keyword_results = keyword_search(query, client, collection_name)

    # Perform semantic search
    semantic_results = search_by_semantic_embedding(query, client, collection_name, threshold)

    # Combine results
    combined_results = keyword_results + semantic_results

    # Remove duplicates and sort by score
    seen_docs = set()
    unique_results = []
    for score, text in combined_results:
        if text not in seen_docs:
            seen_docs.add(text)
            unique_results.append((score, text))

    # Sort documents by score in descending order
    unique_results.sort(reverse=True, key=lambda x: x[0])
    
    # Get the results text with treshold score >= 0.8
    unique_results = [result for result in unique_results if result[0] >= 0.8]

    # Return the top 5 results
    return unique_results[:3]


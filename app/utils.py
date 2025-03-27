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

def create_llm(api_key):
    """Membuat instance LLM dengan API key yang diberikan"""
    if not api_key:
        raise ValueError("API Key is required")

    return ChatOpenAI(
        api_key=api_key,
        model="gpt-4o-mini",
        temperature=0,
        max_retries=2,
        timeout=120,
    )

def create_chain(api_key):
    """Membuat pipeline LLM dengan API key yang diberikan"""
    llm = create_llm(api_key)  # Membuat instance LLM dengan API key

    SYSTEM_CREATE_CHUNK_CONTEXT = """
    You are an advanced Retrieval-Augmented Generation assistant for Legal Understanding.
    Your role is to process document chunks while maintaining context coherence and relevance.
    """
    
    USER_CREATE_CHUNK_CONTEXT = """
    Berikut adalah potongan teks dari dokumen yang perlu diringkas atau dianalisis:
    
    {text}
    """

    # Menyiapkan template prompt
    prompt = ChatPromptTemplate(
        [
            ("system", SYSTEM_CREATE_CHUNK_CONTEXT),
            ("user", USER_CREATE_CHUNK_CONTEXT),
        ]
    )

    # Menghubungkan prompt dengan model LLM
    chain = prompt | llm
    return chain

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

    if not api_key:
        raise ValueError("API Key is required")

    llm = ChatOpenAI(
        api_key=api_key,
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

import streamlit as st

def process_text_in_chunks(client, chunks_list, result, chain):
    """Memproses teks dalam bentuk chunks menggunakan LLM dan menyimpannya ke Typesense."""
    
    all_summaries = []
    previous_context = "-"  # Konteks awal
    end_pos = len(chunks_list)  # Total jumlah chunks

    # Inisialisasi progress bar Streamlit
    progress_text = "Processing chunks. Please wait..."
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, current_chunk in enumerate(chunks_list):
        try:
            # Invoke LLM untuk memproses chunk saat ini
            chunk_context = chain.invoke(
                input={
                    "superheader": result,
                    "previous_context": previous_context,
                    "current_chunk": current_chunk,
                }
            ).content
        except Exception as e:
            st.error(f"⚠️ Error memproses chunk pada posisi {i}: {e}")
            continue

        # Membuat ID unik untuk chunk
        judul = chunks_list[0].get("nomor_peraturan", "unknown_id")  # Gunakan default "unknown_id" jika tidak ada
        chunk_id = f"{judul}_{i+1}"

        # Tambahkan chunk dengan embedding ke Typesense
        try:
            add_chunk_with_embedding(client, chunk_id, result, previous_context)
        except Exception as e:
            st.error(f"⚠️ Gagal menambahkan chunk ke Typesense pada posisi {i}: {e}")
            continue

        # Simpan rangkuman dan perbarui konteks sebelumnya
        all_summaries.append(chunk_context)
        previous_context = chunk_context

        # Perbarui progress bar
        percentage = int(((i + 1) / end_pos) * 100)
        progress_bar.progress((i + 1) / end_pos)
        status_text.text(f"{progress_text} ({percentage}%)")

    # Selesaikan progress bar
    progress_bar.progress(1.0)
    status_text.text("✅ Document processed into chunks successfully!")

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


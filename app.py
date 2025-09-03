from dotenv import load_dotenv
import chromadb
import google.generativeai as genai
import re
import PyPDF2
import streamlit as st
import os


def parse_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text


def chunk_text(text, chunk_size=500):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) < chunk_size:
            current += sentence + " "
        else:
            chunks.append(current.strip())
            current = sentence + " "
    if current:
        chunks.append(current.strip())
    return chunks


st.title("Gemini RAG Demo")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
text_input = st.text_area("Or paste text here")

query = st.text_input("Ask a question about your document:")

document_text = None
if uploaded_file:
    document_text = parse_pdf(uploaded_file)
    st.write("PDF parsed.")
elif text_input:
    document_text = text_input
    st.write("Text input received.")

chunks = []


load_dotenv()

# Gemini API setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
GEMINI_EMBED_MODEL = "models/text-embedding-004"


def generate_embeddings(text, gemini_model=GEMINI_EMBED_MODEL):
    result = genai.embed_content(
        model=gemini_model, content=text
    )
    return result["embedding"]

# ...existing code...


if document_text:
    chunks = chunk_text(document_text)
    st.write(f"Document chunked into {len(chunks)} chunks.")

    # Embed chunks using Gemini API
    embeddings = [generate_embeddings(chunk) for chunk in chunks]
    st.write("Chunks embedded using Gemini text-embedding-004.")

    # ChromaDB setup (local, in-memory)
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="rag_demo")
    # Add embeddings and documents
    ids = [str(i) for i in range(len(chunks))]
    collection.add(
        embeddings=embeddings,
        documents=chunks,
        ids=ids
    )
    st.write("Embeddings stored in ChromaDB.")

    # Store for retrieval
    st.session_state['chunks'] = chunks
    st.session_state['embeddings'] = embeddings
    st.session_state['collection'] = collection

if text_input:
    # Chunk and embed pasted text
    chunks = chunk_text(text_input)
    st.write(f"Text chunked into {len(chunks)} chunks.")

    embeddings = [generate_embeddings(chunk) for chunk in chunks]
    st.write("Chunks embedded using Gemini text-embedding-004.")

    # ChromaDB setup (local, in-memory)
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="rag_demo")
    ids = [str(i) for i in range(len(chunks))]
    collection.add(
        embeddings=embeddings,
        documents=chunks,
        ids=ids
    )
    st.write("Embeddings stored in ChromaDB.")

    st.session_state['chunks'] = chunks
    st.session_state['embeddings'] = embeddings
    st.session_state['collection'] = collection

if query and 'collection' in st.session_state:
    # Embed query using Gemini API
    query_emb = generate_embeddings(query)
    # Retrieve top 3 relevant chunks from ChromaDB
    results = st.session_state['collection'].query(
        query_embeddings=[query_emb],
        n_results=3
    )
    retrieved_chunks = results['documents'][0]
    context = "\n".join(retrieved_chunks)
    st.write("Retrieved context:")
    st.write(context)

    # Gemini API call
    api_key = os.getenv(
        "GEMINI_API_KEY")
    if not api_key:
        st.warning(
            "Please set your Gemini API key in environment variable GEMINI_API_KEY or Streamlit secrets.")
    else:
        genai.configure(api_key=api_key)
        # Use Gemini 2.0 Flash-Lite model for content generation
        model_name = "models/gemini-2.0-flash-lite"
        try:
            model_gemini = genai.GenerativeModel(model_name)
            prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
            response = model_gemini.generate_content(prompt)
            st.write("Gemini Answer:")
            st.write(response.text)
        except Exception as e:
            st.error(f"Error using Gemini 2.0 Flash-Lite: {e}")

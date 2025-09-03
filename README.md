# Gemini RAG Chat App

A simple Retrieval-Augmented Generation (RAG) chat application using Google Gemini API and ChromaDB, built with Streamlit.

## Features
- Upload a PDF or paste text to create a knowledge base
- Ask questions in a chat interface
- Answers are generated using Gemini 2.0 Flash-Lite model with context retrieved from your document
- Embeddings and retrieval powered by ChromaDB (no server required)

## Setup
1. **Clone the repository**
2. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```
3. **Add your Gemini API key to `.env`**
   ```
   GEMINI_API_KEY=your-gemini-api-key-here
   ```
4. **Run the app**
   ```sh
   streamlit run app.py
   ```

## Usage
- Upload a PDF or paste text
- Start chatting with your document using the chat box
- Each answer is generated using Gemini and relevant document context

## Requirements
- Python 3.8+
- See `requirements.txt` for all packages

## Notes
- ChromaDB runs in-memory, no installation or server needed
- Gemini API key required (get from Google AI Studio)

---
Made with ❤️ using Streamlit, Gemini, and ChromaDB.


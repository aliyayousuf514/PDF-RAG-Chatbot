# 📄 AI PDF RAG Chatbot

An AI-powered PDF Question Answering chatbot built with **Python, Streamlit, Sentence Transformers, FAISS, and Google Gemini API**.

The application allows users to upload a PDF document and ask questions about its content. The system uses **Retrieval-Augmented Generation (RAG)** to retrieve the most relevant information from the uploaded document before generating an answer.

---

## 🚀 Live Demo

🔗 **Streamlit App:**  
Add your deployed Streamlit URL here

🔗 **GitHub Repository:**  
Add your GitHub repository URL here

---

## 📌 Project Overview

Reading and searching through large PDF documents manually can be time-consuming.

This project provides an interactive AI chatbot that allows users to:

- 📂 Upload a PDF document
- 📖 Extract text from the PDF
- ✂️ Split the text into smaller chunks
- 🧠 Generate embeddings using Sentence Transformers
- 🔎 Store and search embeddings using FAISS
- 💬 Ask questions about the uploaded document
- 🤖 Generate answers using Google Gemini
- 📚 Retrieve relevant document information before answering
- 🗑️ Clear chat history when needed

The chatbot is designed to answer questions **based on the uploaded PDF context** rather than relying on unrelated external information.

---

# 🧠 How the RAG System Works

The project follows a Retrieval-Augmented Generation architecture.

```text
                📄 PDF Document
                       │
                       ▼
              📖 Text Extraction
                       │
                       ▼
                 ✂️ Chunking
                       │
                       ▼
          🧠 Sentence Transformer
               Embeddings
                       │
                       ▼
                 🔎 FAISS
              Vector Database
                       │
                       │
                User Question
                       │
                       ▼
             Question Embedding
                       │
                       ▼
               FAISS Search
                       │
                       ▼
           Relevant PDF Chunks
                       │
                       ▼
                Gemini API
                       │
                       ▼
              💬 Final Answer

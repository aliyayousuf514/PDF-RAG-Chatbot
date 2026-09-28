import streamlit as st
import os

from dotenv import load_dotenv
from groq import Groq

from utils import (
    load_embedding_model,
    process_pdf,
    create_embeddings,
    create_faiss_index
)

from chatbot import answer_question


# =========================================================
# 1. PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="📄",
    layout="wide"
)


# =========================================================
# 2. TITLE
# =========================================================

st.title("📄 AI PDF Chatbot")

st.write(
    "Upload a PDF document and ask questions. "
    "The chatbot answers using information from your PDF."
)


# =========================================================
# 3. LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)


# ---------------------------------------------------------
# Streamlit Cloud Secrets
# ---------------------------------------------------------

if not GROQ_API_KEY:

    try:

        GROQ_API_KEY = st.secrets[
            "GROQ_API_KEY"
        ]

    except Exception:

        GROQ_API_KEY = None


# =========================================================
# 4. CHECK API KEY
# =========================================================

if not GROQ_API_KEY:

    st.error(
        "Groq API key not found. "
        "Please add GROQ_API_KEY to your .env file "
        "or Streamlit Cloud Secrets."
    )

    st.stop()


# =========================================================
# 5. CREATE GROQ CLIENT
# =========================================================

client = Groq(
    api_key=GROQ_API_KEY
)


# =========================================================
# 6. LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource
def get_embedding_model():

    return load_embedding_model()


with st.spinner(
    "🧠 Loading embedding model..."
):

    embedding_model = get_embedding_model()


# =========================================================
# 7. PDF UPLOADER
# =========================================================

uploaded_file = st.file_uploader(
    "📂 Upload your PDF document",
    type=["pdf"]
)


# =========================================================
# 8. CHECK PDF
# =========================================================

if uploaded_file is None:

    st.info(
        "👆 Please upload a PDF document to start chatting."
    )

    st.stop()


# =========================================================
# 9. PROCESS PDF
# =========================================================

@st.cache_data
def cached_process_pdf(
    pdf_bytes
):

    return process_pdf(
        pdf_bytes
    )


with st.spinner(
    "📖 Reading and processing your PDF..."
):

    (
        number_of_pages,
        chunks
    ) = cached_process_pdf(
        uploaded_file.getvalue()
    )


# =========================================================
# 10. CHECK DOCUMENT
# =========================================================

if not chunks:

    st.error(
        "❌ No readable text was found in this PDF."
    )

    st.stop()


# =========================================================
# 11. CREATE EMBEDDINGS
# =========================================================

@st.cache_data
def cached_create_embeddings(
    chunks
):

    return create_embeddings(
        chunks,
        embedding_model
    )


with st.spinner(
    "🧠 Creating document embeddings..."
):

    embeddings = cached_create_embeddings(
        chunks
    )


# =========================================================
# 12. CREATE FAISS INDEX
# =========================================================

@st.cache_resource
def cached_create_faiss_index(
    embeddings
):

    return create_faiss_index(
        embeddings
    )


with st.spinner(
    "🔎 Creating vector database..."
):

    index = cached_create_faiss_index(
        embeddings
    )


# =========================================================
# 13. INITIALIZE CHAT HISTORY
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# =========================================================
# 14. SIDEBAR
# =========================================================

with st.sidebar:

    st.header(
        "📄 Document Information"
    )

    st.write(
        f"📑 Pages: {number_of_pages}"
    )

    st.write(
        f"🧩 Text Chunks: {len(chunks)}"
    )

    st.write(
        f"📦 FAISS Vectors: {index.ntotal}"
    )

    st.divider()

    st.header(
        "💬 Chat"
    )

    if st.button(
        "🗑️ Clear Chat"
    ):

        st.session_state.messages = []

        st.rerun()


# =========================================================
# 15. DISPLAY PREVIOUS CHAT
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# =========================================================
# 16. CHAT INPUT
# =========================================================

question = st.chat_input(
    "💬 Ask a question about your PDF..."
)


# =========================================================
# 17. PROCESS QUESTION
# =========================================================

if question:

    # -----------------------------------------------------
    # Save user question
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    # -----------------------------------------------------
    # Display user message
    # -----------------------------------------------------

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )


    # -----------------------------------------------------
    # Generate answer
    # -----------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "🔎 Searching your PDF and generating answer..."
        ):

            (
                answer,
                best_distance,
                relevant_chunks
            ) = answer_question(
                question=question,
                embedding_model=embedding_model,
                index=index,
                chunks=chunks,
                client=client
            )


        # =================================================
        # CHECK FOR GROQ API ERROR
        # =================================================

        if answer.startswith("ERROR_STATUS:"):

            # ---------------------------------------------
            # 429 - Rate Limit
            # ---------------------------------------------

            if "ERROR_STATUS: 429" in answer:

                st.error(
                    "🚨 Groq API Rate Limit (429)"
                )

                st.warning(
                    "Too many requests were sent to "
                    "the Groq API. Please wait a little "
                    "while before trying again."
                )


            # ---------------------------------------------
            # 503 - Service Unavailable
            # ---------------------------------------------

            elif "ERROR_STATUS: 503" in answer:

                st.error(
                    "⚠️ Groq Service Unavailable (503)"
                )

                st.warning(
                    "Groq is temporarily unavailable "
                    "or experiencing high demand. "
                    "Please try again later."
                )


            # ---------------------------------------------
            # 500 - Internal Server Error
            # ---------------------------------------------

            elif "ERROR_STATUS: 500" in answer:

                st.error(
                    "❌ Groq Internal Server Error (500)"
                )


            # ---------------------------------------------
            # Unknown error
            # ---------------------------------------------

            else:

                st.error(
                    "❌ Groq API Error"
                )


            # ---------------------------------------------
            # SHOW ACTUAL ERROR DETAILS
            # ---------------------------------------------

            with st.expander(
                "🔍 Show Technical Error Details"
            ):

                st.code(
                    answer,
                    language="text"
                )


        # =================================================
        # NORMAL ANSWER
        # =================================================

        else:

            st.markdown(
                answer
            )


    # -----------------------------------------------------
    # Save answer
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

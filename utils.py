import io
import faiss
import numpy as np

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =========================================================
# 1. LOAD EMBEDDING MODEL
# =========================================================

def load_embedding_model():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model


# =========================================================
# 2. PROCESS PDF
# =========================================================

def process_pdf(pdf_bytes):

    # Convert bytes into a file-like object
    pdf_file = io.BytesIO(pdf_bytes)

    # Read PDF
    pdf_reader = PdfReader(
        pdf_file
    )

    number_of_pages = len(
        pdf_reader.pages
    )

    full_text = ""

    # Extract text from every page
    for page in pdf_reader.pages:

        text = page.extract_text()

        if text:

            full_text += text + "\n"


    # =====================================================
    # Split text into chunks
    # =====================================================

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_text(
        full_text
    )

    return number_of_pages, chunks


# =========================================================
# 3. CREATE EMBEDDINGS
# =========================================================

def create_embeddings(
    chunks,
    embedding_model
):

    embeddings = embedding_model.encode(
        chunks
    )

    return embeddings


# =========================================================
# 4. CREATE FAISS VECTOR DATABASE
# =========================================================

def create_faiss_index(
    embeddings
):

    embeddings_array = np.array(
        embeddings
    ).astype("float32")

    dimension = embeddings_array.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(
        embeddings_array
    )

    return index
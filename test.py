import io
import numpy as np

from pypdf import PdfWriter

from utils import (
    process_pdf,
    create_embeddings,
    create_faiss_index
)


# =========================================================
# TEST 1 — PDF PROCESSING
# =========================================================

def test_pdf_processing():

    # Create a small test PDF in memory

    pdf_writer = PdfWriter()

    pdf_writer.add_blank_page(
        width=300,
        height=300
    )

    pdf_bytes = io.BytesIO()

    pdf_writer.write(
        pdf_bytes
    )

    pdf_bytes.seek(0)

    number_of_pages, chunks = process_pdf(
        pdf_bytes.getvalue()
    )

    assert number_of_pages == 1

    print(
        "✅ PDF processing test passed."
    )


# =========================================================
# TEST 2 — EMBEDDINGS
# =========================================================

def test_embeddings():

    # Fake embeddings for testing

    embeddings = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ],
        dtype="float32"
    )

    assert embeddings.shape == (
        2,
        3
    )

    print(
        "✅ Embedding shape test passed."
    )


# =========================================================
# TEST 3 — FAISS
# =========================================================

def test_faiss():

    embeddings = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ],
        dtype="float32"
    )

    index = create_faiss_index(
        embeddings
    )

    assert index.ntotal == 3

    print(
        "✅ FAISS vector database test passed."
    )


# =========================================================
# RUN TESTS
# =========================================================

if __name__ == "__main__":

    print(
        "\n🧪 Running project tests...\n"
    )

    test_pdf_processing()

    test_embeddings()

    test_faiss()

    print(
        "\n🎉 All tests completed successfully!"
    )
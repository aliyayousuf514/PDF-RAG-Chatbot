import time
import faiss
import numpy as np


# =========================================================
# 1. SEARCH RELEVANT DOCUMENT CHUNKS
# =========================================================

def search_relevant_chunks(
    question,
    embedding_model,
    index,
    chunks,
    k=3
):

    # -----------------------------------------------------
    # Convert question into embedding
    # -----------------------------------------------------

    question_embedding = embedding_model.encode(
        [question]
    )

    question_embedding = np.array(
        question_embedding
    ).astype("float32")


    # -----------------------------------------------------
    # Search FAISS
    # -----------------------------------------------------

    distances, indices = index.search(
        question_embedding,
        k
    )


    # -----------------------------------------------------
    # Get relevant chunks
    # -----------------------------------------------------

    relevant_chunks = []

    for i in indices[0]:

        if i != -1:

            relevant_chunks.append(
                chunks[i]
            )


    # Best distance
    best_distance = distances[0][0]


    return (
        relevant_chunks,
        best_distance,
        distances
    )


# =========================================================
# 2. CREATE RAG PROMPT
# =========================================================

def create_rag_prompt(
    question,
    relevant_chunks
):

    context = "\n\n".join(
        relevant_chunks
    )


    prompt = f"""
You are an AI assistant that answers questions
about an uploaded PDF document.

You MUST follow these rules:

1. Answer ONLY using the document context below.
2. Do NOT use outside knowledge.
3. Do NOT make up information.
4. Do NOT assume information that is not present.
5. If the answer cannot be found in the document,
   respond exactly:

"I could not find this information in the uploaded document."

6. Keep the answer clear and easy to understand.
7. Stay directly focused on the user's question.

DOCUMENT CONTEXT:
================================

{context}

================================

USER QUESTION:

{question}

ANSWER:
"""

    return prompt


# =========================================================
# 3. ASK GEMINI
# =========================================================

def ask_gemini(
    client,
    prompt,
    max_retries=4
):

    response = None
    answer = None

    for attempt in range(
        max_retries
    ):

        try:

            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt
            )

            answer = response.text

            return answer


        except Exception as e:

            error_message = str(e)


            # -------------------------------------------------
            # Handle 503 error
            # -------------------------------------------------

            if "503" in error_message:

                if attempt < max_retries - 1:

                    wait_time = 2 ** attempt

                    time.sleep(
                        wait_time
                    )

                else:

                    return (
                        "Gemini is currently experiencing "
                        "high demand. Please try again later."
                    )


            else:

                return (
                    f"Gemini API error: {e}"
                )


    return (
        "Unable to generate an answer."
    )


# =========================================================
# 4. COMPLETE RAG QUESTION FUNCTION
# =========================================================

def answer_question(
    question,
    embedding_model,
    index,
    chunks,
    client,
    relevance_threshold=1.20
):

    # -----------------------------------------------------
    # Search document
    # -----------------------------------------------------

    (
        relevant_chunks,
        best_distance,
        distances
    ) = search_relevant_chunks(
        question=question,
        embedding_model=embedding_model,
        index=index,
        chunks=chunks,
        k=3
    )


    # -----------------------------------------------------
    # Check whether question is relevant
    # -----------------------------------------------------

    if best_distance > relevance_threshold:

        return (
            "I could not find this information "
            "in the uploaded document.",
            best_distance,
            relevant_chunks
        )


    # -----------------------------------------------------
    # Create RAG prompt
    # -----------------------------------------------------

    prompt = create_rag_prompt(
        question,
        relevant_chunks
    )


    # -----------------------------------------------------
    # Ask Gemini
    # -----------------------------------------------------

    answer = ask_gemini(
        client,
        prompt
    )


    return (
        answer,
        best_distance,
        relevant_chunks
    )
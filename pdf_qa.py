import csv
import traceback
from pathlib import Path

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction


# -------------------- CONFIGURATION --------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"
# Accept either filename; keep the uploaded book PDF inside dataset/
PDF_CANDIDATES = [DATASET_DIR / "sample.pdf", DATASET_DIR / "sample(1).pdf"]
PDF_PATH = next((candidate for candidate in PDF_CANDIDATES if candidate.exists()), PDF_CANDIDATES[0])

EXTRACTED_TEXT_PATH = DATASET_DIR / "extracted_text.txt"
CHUNKS_CSV_PATH = DATASET_DIR / "chunks.csv"
QUESTIONS_CSV_PATH = DATASET_DIR / "questions.csv"
SEARCH_RESULTS_CSV_PATH = DATASET_DIR / "search_results.csv"
COMPARISON_CSV_PATH = DATASET_DIR / "answers_comparison.csv"
SUMMARY_PATH = PROJECT_DIR / "run_summary.txt"

CHROMA_PATH = PROJECT_DIR / "chroma_db"
COLLECTION_NAME = "week13_pdf_qa"

OLLAMA_MODEL = "llama3.2"
TOP_K = 5
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


QUESTIONS = [
    "How does the book define Artificial Intelligence, and what tasks are associated with human intelligence?",
    "How did AI evolve from Symbolic AI to Machine Learning, Deep Learning, Generative AI, and Large Language Models?",
    "What is Machine Learning, and how is learning from data different from traditional rule-based programming?",
    "What are the differences between supervised, unsupervised, and reinforcement learning?",
    "Why are data quality, representativeness, and training, validation, and test datasets important in Machine Learning?",
    "What is a neural network, and what are the roles of input, hidden, and output layers?",
    "What are tokens, and why does tokenization matter for Large Language Models?",
    "What are embeddings, and how do they represent meaning in numerical form?",
    "What is the Transformer architecture, and how does self-attention help process token relationships?",
    "How are Large Language Models trained using next-token prediction, loss functions, backpropagation, and gradient descent?",
    "How does an LLM answer a user request during inference, from tokenization to next-token prediction?",
    "What are the capabilities and limitations of LLMs, including hallucinations, reasoning, code generation, and language generation?",
    "What are common applications of LLMs across education, software engineering, healthcare, business, and customer service?",
    "What is Retrieval-Augmented Generation (RAG), and how can it integrate LLMs with information systems and databases?",
    "What privacy, security, safety, ethical, governance, and human-oversight concerns does the book discuss for LLM-based systems?",
]


def write_csv(path: Path, headers: list, rows: list):
    """Write rows into a CSV file with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)


def extract_pdf_text():
    print("\n[1/7] Extracting text from PDF...")

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF file not found: {PDF_PATH}\n"
            "Please place sample.pdf or sample(1).pdf inside the project's dataset folder."
        )

    reader = PdfReader(str(PDF_PATH))
    all_text = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        all_text.append(f"\n--- Page {page_number} ---\n{page_text}")

    extracted_text = "\n".join(all_text).strip()

    if not extracted_text:
        raise ValueError(
            "No text was extracted from the PDF. The PDF may be scanned/image-only."
        )

    EXTRACTED_TEXT_PATH.write_text(extracted_text, encoding="utf-8")
    print(f"PDF pages: {len(reader.pages)}")
    print(f"Characters extracted: {len(extracted_text)}")
    print(f"Saved: {EXTRACTED_TEXT_PATH}")
    return extracted_text


def create_chunks(extracted_text: str):
    print("\n[2/7] Splitting PDF text into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_text(extracted_text)

    chunk_rows = [
        [f"chunk_{index:04d}", chunk]
        for index, chunk in enumerate(chunks, start=1)
    ]

    write_csv(CHUNKS_CSV_PATH, ["chunk_id", "chunk_text"], chunk_rows)

    print(f"Total chunks: {len(chunks)}")
    print(f"Saved: {CHUNKS_CSV_PATH}")
    return chunks


def create_vector_database(chunks):
    print("\n[3/7] Creating/updating ChromaDB vector database...")

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    embedding_function = DefaultEmbeddingFunction()

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"description": "Week 13 PDF Question Answering"},
    )

    ids = [f"pdf_chunk_{index:04d}" for index in range(1, len(chunks) + 1)]
    metadatas = [{"chunk_number": index} for index in range(1, len(chunks) + 1)]

    # Upsert means rerunning the program updates matching IDs instead of duplicating them.
    collection.upsert(
        ids=ids,
        documents=chunks,
        metadatas=metadatas,
    )

    print(f"Collection: {COLLECTION_NAME}")
    print(f"Chunks stored: {collection.count()}")
    print(f"Database folder: {CHROMA_PATH}")
    return collection


def retrieve_documents(collection, question: str):
    result = collection.query(
        query_texts=[question],
        n_results=min(TOP_K, max(collection.count(), 1)),
        include=["documents", "metadatas", "distances"],
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    retrieved = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        retrieved.append(
            {
                "document": document,
                "chunk_number": metadata.get("chunk_number", ""),
                "distance": distance,
            }
        )

    return retrieved


def ask_ollama(llm, question: str, context: str = None):
    if context:
        prompt = f"""
You are a helpful question-answering assistant.
Answer the user's question using only the context given below.
If the context does not contain the answer, clearly say:
"The provided PDF does not contain enough information to answer this question."
Keep the answer clear and student-friendly.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
    else:
        prompt = f"""
You are a helpful AI assistant. Answer the following question using your general
knowledge. Keep the answer clear and student-friendly.

QUESTION:
{question}

ANSWER:
"""

    response = llm.invoke(prompt)
    return response.content.strip()


def run_question_answering(collection):
    print("\n[4/7] Checking Ollama model and preparing question-answering...")

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0,
    )

    search_rows = []
    comparison_rows = []

    print("\n[5/7] Running 15 questions (RAG and without RAG)...")

    for index, question in enumerate(QUESTIONS, start=1):
        print(f"\nQuestion {index}/{len(QUESTIONS)}: {question}")

        retrieved = retrieve_documents(collection, question)

        context = "\n\n".join(
            item["document"] for item in retrieved
        )

        rag_answer = ask_ollama(llm, question, context=context)
        no_rag_answer = ask_ollama(llm, question, context=None)

        for rank, item in enumerate(retrieved, start=1):
            similarity_score = 1 / (1 + float(item["distance"]))
            search_rows.append(
                [
                    index,
                    question,
                    rank,
                    item["chunk_number"],
                    item["distance"],
                    similarity_score,
                    item["document"],
                ]
            )

        comparison_rows.append(
            [index, question, rag_answer, no_rag_answer]
        )

        print("RAG answer:", rag_answer[:300].replace("\n", " "), "...")
        print("Without RAG answer:", no_rag_answer[:200].replace("\n", " "), "...")

    write_csv(
        SEARCH_RESULTS_CSV_PATH,
        [
            "question_id",
            "question",
            "rank",
            "chunk_number",
            "distance",
            "derived_similarity_score",
            "retrieved_text",
        ],
        search_rows,
    )

    write_csv(
        COMPARISON_CSV_PATH,
        ["question_id", "question", "rag_answer", "without_rag_answer"],
        comparison_rows,
    )

    print(f"Saved: {SEARCH_RESULTS_CSV_PATH}")
    print(f"Saved: {COMPARISON_CSV_PATH}")
    return comparison_rows


def create_summary(extracted_text, chunks, comparison_rows, collection):
    print("\n[6/7] Creating project summary...")

    summary = f"""WEEK 13 - PDF QUESTION ANSWERING SYSTEM
=====================================

Project: PDF Question Answering System using LangChain, ChromaDB and Ollama

PDF file: {PDF_PATH}
PDF text characters: {len(extracted_text)}
Number of chunks: {len(chunks)}
Questions processed: {len(comparison_rows)}
Top-K retrieval: {TOP_K}
Chunk size: {CHUNK_SIZE}
Chunk overlap: {CHUNK_OVERLAP}
Ollama model: {OLLAMA_MODEL}
ChromaDB collection: {COLLECTION_NAME}
ChromaDB stored records: {collection.count()}

GENERATED FILES
---------------
dataset/extracted_text.txt
dataset/chunks.csv
dataset/questions.csv
dataset/search_results.csv
dataset/answers_comparison.csv
chroma_db/ (persistent vector database)

NOTES
-----
- RAG answers are generated using retrieved PDF chunks as context.
- Without-RAG answers are generated using the Ollama model's general knowledge.
- Similarity score is derived from Chroma distance using 1 / (1 + distance).
  It is a derived value and not a probability.
"""

    SUMMARY_PATH.write_text(summary, encoding="utf-8")
    print(f"Saved: {SUMMARY_PATH}")


def main():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    write_csv(
        QUESTIONS_CSV_PATH,
        ["question_id", "question"],
        [[index, question] for index, question in enumerate(QUESTIONS, start=1)],
    )
    print(f"Questions CSV created: {QUESTIONS_CSV_PATH}")

    extracted_text = extract_pdf_text()
    chunks = create_chunks(extracted_text)
    collection = create_vector_database(chunks)
    comparison_rows = run_question_answering(collection)
    create_summary(extracted_text, chunks, comparison_rows, collection)

    print("\n[7/7] WEEK 13 PROJECT COMPLETED!")
    print("\nGenerated files:")
    print(f"- {EXTRACTED_TEXT_PATH}")
    print(f"- {CHUNKS_CSV_PATH}")
    print(f"- {QUESTIONS_CSV_PATH}")
    print(f"- {SEARCH_RESULTS_CSV_PATH}")
    print(f"- {COMPARISON_CSV_PATH}")
    print(f"- {SUMMARY_PATH}")
    print(f"- {CHROMA_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nPROJECT ERROR:")
        print(error)
        print("\nDetailed traceback:")
        traceback.print_exc()
        print(
            "\nPlease copy and send the error output so it can be fixed."
        )

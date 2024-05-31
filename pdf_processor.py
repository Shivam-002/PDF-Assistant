from PyPDF2 import PdfReader
import os

from fastapi import HTTPException

from embedding_processor import (
    text_splitter,
    get_embeddings,
    save_embeddings,
    save_chunks,
    load_embeddings,
    load_chunks,
    create_faiss_index,
    similarity_search,
    embeddings_exists,
)

PDF_FOLDER = "data/pdfs"

os.makedirs(PDF_FOLDER, exist_ok=True)


def read_pdf(filename):
    """Read the PDF file and extract text."""
    try:
        full_path = f"{PDF_FOLDER}/{filename}"
        pdf = PdfReader(open(full_path, "rb"))
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        return text
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {filename}.")


def upload_pdf(file):

    if not is_pdf(file.filename):
        raise HTTPException(status_code=400, detail="File is not a PDF.")

    if not pdf_exists(file.filename):
        print(f"Saving {file.filename}")
        save_pdf(file)
    else:
        print(f"{file.filename} already exists.")


def save_pdf(file):
    contents = file.file.read()
    with open(f"{PDF_FOLDER}/{file.filename}", "wb") as f:
        f.write(contents)
    file.file.close()


def process_pdf(filename):
    """Main process to embed and save PDF chunks."""

    if not is_pdf(filename):
        raise HTTPException(status_code=400, detail=f"{filename} is not a PDF.")

    embedding_filename = f"{filename}_embeddings.pkl"

    if embeddings_exists(embedding_filename):
        return

    text = read_pdf(filename)
    chunks = text_splitter(text)
    embeddings = get_embeddings(chunks)
    save_embeddings(filename, embeddings)
    save_chunks(filename, chunks)


def query_pdf(filename, query):
    """Query function to search for similar chunks."""
    embeddings = load_embeddings(filename)
    chunks = load_chunks(filename)
    index = create_faiss_index(embeddings)
    similar_indices, similar_distances = similarity_search(index, query)
    similar_chunks = [chunks[i] for i in similar_indices]
    return similar_chunks, similar_distances


def pdf_exists(filename):
    return os.path.exists(PDF_FOLDER + "/" + filename)


def is_pdf(filename):
    """Check if the file is a PDF."""
    return filename.lower().endswith(".pdf")

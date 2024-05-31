import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pickle
from transformers import AutoModel, AutoTokenizer
import torch
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import google.generativeai as genai
from utils import template, CHUNK_SIZE, OVERLAP_SIZE

EMBEDDING_FOLDER = "data/embeddings"
os.makedirs(EMBEDDING_FOLDER, exist_ok=True)

load_dotenv()

model_name = "intfloat/multilingual-e5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def text_splitter(text):
    """Split the text into chunks."""
    print("text_splitter")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=OVERLAP_SIZE,
    )
    return splitter.split_text(text)


def get_embeddings(text_list):
    """Generate embeddings for a list of texts."""
    print("get_embeddings")
    inputs = tokenizer(text_list, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    pooled_embeddings = torch.mean(embeddings, dim=1)
    return pooled_embeddings


def save_embeddings(filename, embeddings):
    """Save embeddings to a file."""
    print("save_embeddings")
    with open(os.path.join(EMBEDDING_FOLDER, filename + "_embeddings.pkl"), "wb") as f:
        pickle.dump(embeddings, f)


def save_chunks(filename, chunks):
    """Save chunks to a file."""
    print("save_chunks")
    with open(os.path.join(EMBEDDING_FOLDER, filename + "_chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)


def load_embeddings(filename):
    """Load embeddings from a file."""
    print("load_embeddings")
    with open(os.path.join(EMBEDDING_FOLDER, filename + "_embeddings.pkl"), "rb") as f:
        embeddings = pickle.load(f)
    return embeddings


def load_chunks(filename):
    """Load chunks from a file."""
    print("load_chunks")
    with open(os.path.join(EMBEDDING_FOLDER, filename + "_chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)
    return chunks


def create_faiss_index(embeddings):
    """Create a FAISS index and add embeddings to it."""
    print("create_faiss_index")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def similarity_search(index, query_text, top_k=3):
    """Perform a similarity search on the FAISS index."""
    print("similarity_search")
    query_embedding = get_embeddings([query_text]).numpy()
    distances, indices = index.search(query_embedding, top_k)
    return indices[0], distances[0]


def embeddings_exists(filename):
    return os.path.exists(EMBEDDING_FOLDER + "/" + filename)

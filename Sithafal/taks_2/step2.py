import os
from bs4 import BeautifulSoup
import requests
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from dotenv import load_dotenv
from transformers import pipeline

# Load Hugging Face API key from environment variable
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
if not HUGGINGFACE_API_KEY:
    raise ValueError("Hugging Face API key not found. Please set it in the environment variables or .env file.")

# Hugging Face model pipeline
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", use_auth_token=HUGGINGFACE_API_KEY)


# Scraping and Preprocessing
def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text(separator=' ')
    return text


def chunk_text(text, chunk_size=500):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


# Embedding Generation
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def generate_embeddings(chunks):
    embeddings = model.encode(chunks, convert_to_tensor=True)
    return embeddings.cpu().detach().numpy()  # Convert to NumPy array


# Storing Embeddings in FAISS
def store_embeddings(embeddings, metadata):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, metadata


# Query Embedding
def query_to_embedding(query):
    embedding = model.encode([query], convert_to_tensor=True)
    return embedding.cpu().detach().numpy()  # Convert to NumPy array


# Similarity Search
def similarity_search(query_embedding, index, metadata, k=5):
    query_embedding = np.array(query_embedding, dtype='float32')  # Ensure it's a NumPy array

    # Perform search
    distances, indices = index.search(query_embedding, k)

    # Filter results to avoid out-of-bounds errors
    results = []
    for idx, i in enumerate(indices[0]):
        if 0 <= i < len(metadata):
            results.append((metadata[i], distances[0][idx]))
        else:
            print(f"Warning: Index {i} is out of bounds for metadata.")
    return results


# Response Generation using Hugging Face
def generate_response(query, retrieved_chunks):
    context = " ".join([chunk[0] for chunk in retrieved_chunks])
    result = qa_pipeline(question=query, context=context)
    return result['answer']


# Data Ingestion
urls = [
    "https://www.uchicago.edu/",
    "https://www.washington.edu/",
    "https://www.stanford.edu/",
    "https://und.edu/"
]

texts = [scrape_website(url) for url in urls]
chunks = [chunk for text in texts for chunk in chunk_text(text)]
embeddings = generate_embeddings(chunks)
index, metadata = store_embeddings(embeddings, chunks)

# Validate FAISS index size
if index.ntotal != len(metadata):
    raise ValueError("Mismatch between FAISS index size and metadata size.")

# Query Handling
query = "What programs does Stanford offer?"
query_embedding = query_to_embedding(query)

# Debugging Information
print("Index total entries:", index.ntotal)
print("Query embedding shape:", query_embedding.shape)

# Perform Similarity Search
relevant_chunks = similarity_search(query_embedding, index, metadata)

if not relevant_chunks:
    print("No relevant chunks found.")
else:
    for chunk, distance in relevant_chunks:
        print(f"Relevant Chunk: {chunk[:100]}... (Distance: {distance})")

    response = generate_response(query, relevant_chunks)
    print("Generated Response:")
    print(response)

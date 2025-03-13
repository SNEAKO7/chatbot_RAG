import os
import PyPDF2
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter

def load_file_texts(folder_path):
    """
    Reads text from all PDFs and TXT files in the given folder.
    Returns a list of text strings (one per file).
    """
    texts = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if filename.lower().endswith('.pdf'):
            try:
                with open(filepath, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    if text:
                        texts.append(text)
            except Exception as e:
                print(f"Error reading PDF {filename}: {e}")
        elif filename.lower().endswith('.txt'):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                    texts.append(text)
            except Exception as e:
                print(f"Error reading TXT {filename}: {e}")
    return texts

def retrieve_context(query, folder_path, k=3):
    """
    Loads all PDFs and TXT files from folder_path, splits their text into chunks,
    builds a FAISS index using sentence-transformer embeddings, and retrieves the
    top k relevant chunks for the query.
    """
    texts = load_file_texts(folder_path)
    if not texts:
        return "No documents found in the folder."

    # Split each document into chunks.
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for text in texts:
        chunks.extend(text_splitter.split_text(text))

    # Create embeddings and build FAISS index.
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Retrieve the top k relevant chunks.
    docs = vector_store.similarity_search(query, k=k)
    context = "\n".join([doc.page_content for doc in docs])
    return context

''' #UPDATED FOR MISTRAL-7B
'''✔ Mistral-7B (8192 tokens) → No more token overflow issues!
✔ Better RAG context retrieval using HuggingFaceEmbeddings
✔ Recursive text splitting for improved chunking
✔ Stronger error handling'''
'''
import os
import PyPDF2
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
# Load text from PDFs and TXT files
def load_file_texts(folder_path):
    texts = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)

        if filename.lower().endswith('.pdf'):
            try:
                with open(filepath, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
                    if text:
                        texts.append(text)
            except Exception as e:
                print(f"Error reading PDF {filename}: {e}")

        elif filename.lower().endswith('.txt'):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                    texts.append(text)
            except Exception as e:
                print(f"Error reading TXT {filename}: {e}")
    
    return texts

# Retrieve relevant context
def retrieve_context(query, folder_path, k=3):
    texts = load_file_texts(folder_path)
    if not texts:
        return "No relevant documents found."

    # Split text into meaningful chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = [chunk for text in texts for chunk in text_splitter.split_text(text)]

    # Use HuggingFace embeddings instead of old SentenceTransformerEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Retrieve top k relevant chunks
    docs = vector_store.similarity_search(query, k=k)
    context = "\n".join([doc.page_content for doc in docs])
    
    return context if context else "No relevant information found in documents."
    
#Instead of rebuilding the FAISS index on every query, save it and reload it when needed.


import os
import PyPDF2
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle

VECTOR_STORE_PATH = "faiss_index"

# Load text from PDFs and TXT files
def load_file_texts(folder_path):
    texts = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)

        if filename.lower().endswith('.pdf'):
            try:
                with open(filepath, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
                    if text:
                        texts.append(text)
            except Exception as e:
                print(f"Error reading PDF {filename}: {e}")

        elif filename.lower().endswith('.txt'):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                    texts.append(text)
            except Exception as e:
                print(f"Error reading TXT {filename}: {e}")
    
    return texts

# Load or create FAISS vector store
def get_vector_store(texts):
    if os.path.exists(VECTOR_STORE_PATH):
        # Load existing FAISS index
        with open(VECTOR_STORE_PATH, "rb") as f:
            return pickle.load(f)
    else:
        # Create new FAISS index
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = [chunk for text in texts for chunk in text_splitter.split_text(text)]
        
        vector_store = FAISS.from_texts(chunks, embeddings)

        # Save FAISS index for reuse
        with open(VECTOR_STORE_PATH, "wb") as f:
            pickle.dump(vector_store, f)

        return vector_store

# Retrieve relevant context
def retrieve_context(query, folder_path, k=3):
    texts = load_file_texts(folder_path)
    if not texts:
        return "No relevant documents found."

    vector_store = get_vector_store(texts)
    docs = vector_store.similarity_search(query, k=k)
    context = "\n".join([doc.page_content for doc in docs])

    return context if context else "No relevant information found in documents."'''
'''#fine tuning the intro and outro messages
import os
import PyPDF2
import warnings
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ Corrected import
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore", category=UserWarning)

# Load text from PDFs and TXT files
def load_file_texts(folder_path):
    texts = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)

        if filename.lower().endswith('.pdf'):
            try:
                with open(filepath, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
                    if text:
                        texts.append(text)
            except Exception as e:
                print(f"Error reading PDF {filename}: {e}")

        elif filename.lower().endswith('.txt'):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                    texts.append(text)
            except Exception as e:
                print(f"Error reading TXT {filename}: {e}")
    
    return texts

# Global cache for the vector store
vector_store_cache = None

def get_vector_store(texts):
    global vector_store_cache
    if vector_store_cache is not None:
        return vector_store_cache  # Use cached vector store

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = [chunk for text in texts for chunk in text_splitter.split_text(text)]
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store_cache = FAISS.from_texts(chunks, embeddings)
    
    return vector_store_cache

# Retrieve relevant context
def retrieve_context(query, folder_path, k=3):
    texts = load_file_texts(folder_path)
    if not texts:
        return "Cal: No relevant documents found."

    vector_store = get_vector_store(texts)
    docs = vector_store.similarity_search(query, k=k)
    
    extracted_texts = [doc.page_content.strip() for doc in docs]
    return f"Cal: {extracted_texts[0]}" if extracted_texts else "Cal: No relevant information found."

# Main chatbot function
def chat():
    print("\nCal: Hi! I'm Cal, the Callippus assistant. I help retrieve information from FRS and UAT documents.\n")

    folder_path = "D:\\chatbot_RAG\\data"  # ✅ Updated to your actual document path
    
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit", "bye"]:
            print("Cal: Goodbye! If you need help again, just ask. Take care! 😊")
            break
        
        response = retrieve_context(query, folder_path)
        print(response)

# Run the chatbot
if __name__ == "__main__":
    chat()'''

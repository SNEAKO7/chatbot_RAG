'''import os
import PyPDF2
from langchain_huggingface import HuggingFaceEmbeddings
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

    # Split each document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for text in texts:
        chunks.extend(text_splitter.split_text(text))

    # Create embeddings using HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create FAISS index
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Retrieve the top k relevant chunks
    docs = vector_store.similarity_search(query, k=k)
    context = "\n".join([doc.page_content for doc in docs])
    
    return context if context else "No relevant context found."'''
#UPDATED FOR MISTRAL-7B
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
    
    return context if context else "No relevant information found in documents."'''
#Instead of rebuilding the FAISS index on every query, save it and reload it when needed.
'''
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

    return context if context else "No relevant information found in documents."
'''
#BETTER CONETXT RETRIEVAL AND CHUNKING
'''import os
import PyPDF2
import warnings
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ Updated import
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

# Global cache for the vector store to avoid recomputation
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
        return "No relevant documents found."

    vector_store = get_vector_store(texts)
    docs = vector_store.similarity_search(query, k=k)
    
    extracted_texts = [doc.page_content.strip() for doc in docs]
    return extracted_texts[0] if extracted_texts else "No relevant information found."
'''
#fine tuning the intro and outro messages
'''import os
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

#update for hallucinating
import os
import PyPDF2
import warnings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_cpp import Llama  

warnings.filterwarnings("ignore", category=UserWarning)

# Memory storage for follow-up questions
chat_memory = {}

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
        return vector_store_cache  

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = [chunk for text in texts for chunk in text_splitter.split_text(text)]
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store_cache = FAISS.from_texts(chunks, embeddings)
    
    return vector_store_cache

# Initialize Llama model for fallback answers
model_path = "D:/chatbot_RAG/llama.cpp/models/mistral-7b-instruct-v0.1-q4_k_m.gguf"
llm = Llama(model_path=model_path, n_ctx=8192, n_batch=512, verbose=False)

# Retrieve relevant context
def retrieve_context(query, folder_path, k=3):
    global chat_memory  # Track memory for follow-up questions

    # Check if it's a follow-up question
    if query.lower().startswith("is") or query.lower().startswith("what is"):
        for key in chat_memory:
            if key in query.lower():
                return f"Cal: {chat_memory[key]}"

    texts = load_file_texts(folder_path)
    if not texts:
        return "Cal: No relevant documents found."

    vector_store = get_vector_store(texts)
    docs = vector_store.similarity_search(query, k=k)
    
    extracted_texts = [doc.page_content.strip() for doc in docs]
    
    if extracted_texts:
        answer = extracted_texts[0]
        chat_memory[query.lower()] = answer  # Save answer for follow-ups
        return f"Cal: {answer}"  
    else:
        # No answer found in documents
        fallback_response = get_fallback_response(query)
        chat_memory[query.lower()] = fallback_response  # Store fallback answer
        return f"Cal: The context does not provide enough details, but here is what I found:\n\n{fallback_response}"

# Use LLM for general responses when answer is missing
def get_fallback_response(query):
    prompt = f"""
    You are an AI assistant. If the answer is not in the documents, provide a general answer.

    Question: {query}
    Answer:
    """
    response = llm(prompt, temperature=0.2, max_tokens=100)
    return response["choices"][0]["text"].strip()

# Main chatbot function
def chat():
    print("\nCal: Hi! I'm Cal, the Callippus assistant. I help retrieve information from FRS and UAT documents.\n")

    folder_path = "D:\\chatbot_RAG\\data"  
    
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit", "bye"]:
            print("Cal: Goodbye! If you need help again, just ask. Take care! 😊")
            break
        
        response = retrieve_context(query, folder_path)
        print(response)

# Run the chatbot
if __name__ == "__main__":
    chat()

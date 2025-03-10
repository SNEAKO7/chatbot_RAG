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



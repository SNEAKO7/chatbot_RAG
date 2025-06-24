import os
import re
import json
import PyPDF2
import pandas as pd
import pytesseract
from PIL import Image
import warnings
from docx import Document as DocxDocument
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Tuple

warnings.filterwarnings("ignore", category=UserWarning)

_vector_store_cache = None
_source_chunks = []

# WhatsApp parsing logic
def preprocess_whatsapp_chat(text: str) -> list[str]:
    pattern = r"^\d{1,2}/\d{1,2}/\d{2}, \d{1,2}:\d{2}"
    messages = []
    current = ""

    for line in text.splitlines():
        if re.match(pattern, line):
            if current:
                messages.append(current.strip())
            current = line
        else:
            current += " " + line

    if current:
        messages.append(current.strip())

    # Clean and label messages
    clean_msgs = []
    for msg in messages:
        if "<Media omitted>" in msg or re.fullmatch(r"[👆🏼⬆️]+", msg.strip()):
            continue
        clean_msgs.append(msg)
    return clean_msgs

def load_file_texts(folder_path: str) -> List[Tuple[str, str]]:
    texts = []
    for fn in os.listdir(folder_path):
        path = os.path.join(folder_path, fn)
        low = fn.lower()

        try:
            if low.endswith(".pdf"):
                with open(path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    pages = [p.extract_text() or "" for p in reader.pages]
                    texts.append((fn, "\n".join(pages)))

            elif low.endswith(".docx"):
                doc = DocxDocument(path)
                paras = [para.text for para in doc.paragraphs if para.text.strip()]
                texts.append((fn, "\n".join(paras)))

            elif low.endswith(".txt"):
                raw = open(path, "r", encoding="utf-8").read()
                if " - " in raw and re.match(r"^\d{1,2}/\d{1,2}/\d{2}", raw.strip()):
                    msgs = preprocess_whatsapp_chat(raw)
                    for m in msgs:
                        texts.append((fn, m))
                else:
                    texts.append((fn, raw))

            elif low.endswith(".json"):
                data = json.load(open(path, "r", encoding="utf-8"))
                if isinstance(data, dict):
                    texts.append((fn, json.dumps(data)))
                elif isinstance(data, list):
                    for item in data:
                        texts.append((fn, json.dumps(item)))

            elif low.endswith((".xls", ".xlsx")):
                df_dict = pd.read_excel(path, sheet_name=None)
                for sheet, df in df_dict.items():
                    texts.append((f"{fn} [{sheet}]", df.to_csv(index=False)))

            elif low.endswith((".png", ".jpg", ".jpeg", ".tiff")):
                ocr = pytesseract.image_to_string(Image.open(path))
                texts.append((fn, ocr))

        except Exception as e:
            print(f"Error loading {fn}: {e}")
    return texts

def get_vector_store(texts: List[Tuple[str, str]]) -> FAISS:
    global _vector_store_cache, _source_chunks
    if _vector_store_cache:
        return _vector_store_cache

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks, metadatas = [], []

    for filename, content in texts:
        for chunk in splitter.split_text(content):
            chunks.append(chunk)
            metadatas.append({"source": filename})

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    _vector_store_cache = FAISS.from_texts(chunks, embeddings, metadatas=metadatas)
    _source_chunks = list(zip(chunks, metadatas))
    return _vector_store_cache

def retrieve_context(query: str, folder_path: str, k: int = 7) -> Tuple[str, List[Tuple[str, str]]]:
    texts = load_file_texts(folder_path)
    if not texts:
        return "", []

    vs = get_vector_store(texts)
    docs_and_scores = vs.similarity_search_with_score(query, k=k)

    relevant_chunks = []
    for doc, score in docs_and_scores:
        source = doc.metadata.get("source", "Unknown")
        relevant_chunks.append((doc.page_content, source))

    # Format nicely for readability
    formatted = [
        f"[{source}]\n{chunk.strip()}"
        for chunk, source in relevant_chunks
    ]
    return "\n\n---\n\n".join(formatted), relevant_chunks

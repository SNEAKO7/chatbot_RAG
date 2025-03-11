**Chatbot_RAG**

Chatbot_RAG is a local Retrieval-Augmented Generation (RAG) chatbot that leverages a DeepSeek language model to provide context-aware responses. By integrating a vector-based retrieval pipeline with the DeepSeek model, the chatbot is able to supplement its answers with information extracted from your custom documents (PDFs and TXT files). This project uses the llama-cpp-python bindings to run the DeepSeek model locally and employs FAISS along with LangChain for efficient similarity search.


**Table of Contents**

-Overview

-Features

-What is RAG?

-DeepSeek Model

-Project Structure

-Setup and Installation

-How to Run

-Data Store

-Usage

-Contributing

-License

-Downloading the DeepSeek Model & Installing Ollama

-What is llama.cpp and Why Do We Need It?

-Steps to Download and Set Up llama.cpp


**Features**

Local DeepSeek Model: Runs the deepseek-r1-distill-qwen-7b-q4_k_m.gguf model locally using llama-cpp-python.
Retrieval-Augmented Generation (RAG): Fetches relevant context from PDFs and TXT files to inform the chatbot’s answers.
Vector Store: Uses FAISS with sentence-transformer embeddings via LangChain for efficient similarity search.
Easy Setup & Execution: Simple Python scripts to set up and run the chatbot on your local machine.Chatbot_RAG combines
a local DeepSeek language model with a RAG pipeline. The RAG approach retrieves relevant context from your documents to
augment the chatbot’s generated responses. This allows for more accurate and context-sensitive answers—especially when
dealing with specialized or custom data.


*What is RAG?*

Retrieval-Augmented Generation (RAG) is a technique that enhances language model responses by incorporating external, dynamically retrieved information. Instead of relying solely on its internal (pretrained) knowledge, the model is provided with context retrieved from a datastore (such as a collection of documents), resulting in responses that are more informed and accurate.


*DeepSeek Model*

This project uses the deepseek-r1-distill-qwen-7b-q4_k_m.gguf model—a distilled variant of the DeepSeek language model. This model strikes a balance between performance and accuracy and is run locally via llama-cpp-python.


**Project Structure**

chatbot_RAG/
├── chatbot.py         # Main chatbot script integrating DeepSeek and RAG
├── rag.py             # Retrieval pipeline: extracts text from documents, builds FAISS index, and retrieves context
├── llama.cpp/         # Directory for the DeepSeek model and related files
│   └── models/        # Contains the DeepSeek GGUF model file(s)
├── data/              # Folder for PDFs and TXT files (your document data store)
├── venv/              # Virtual environment (should not be tracked in Git)
├── .gitignore         # Git ignore file to exclude unnecessary files (e.g., venv, large models)
└── README.md          # This README file


**Setup and Installation**
*Prerequisites*

Python 3.10+

Git (or GitHub Desktop)

Virtual Environment (venv)

Git LFS (if you need to track large files; however, it is recommended to exclude venv and model files)

**Installation Steps**

*Clone the Repository:*

-(Open a command prompt or terminal) run:  git clone https://github.com/SNEAKO7/chatbot_RAG.git

                                           cd chatbot_RAG

										  
*Create and Activate a Virtual Environment:*

-Windows (Command Prompt): python -m venv venv

                          venv\Scripts\activate

						  
-Windows (PowerShell):    python -m venv venv

                          .\venv\Scripts\Activate.ps1

		
-*Install Dependencies:*

You can generate a requirements.txt (if not provided) with:

pip freeze >   requirements.txt


Or install the needed packages manually:  pip install llama-cpp-python PyPDF2 langchain faiss-cpu sentence-transformers git+https://github.com/langchain-ai/langchain-community.git


*Set Up Model and Data:* (how to get model explained in the end)

Place your DeepSeek model file (deepseek-r1-distill-qwen-7b-q4_k_m.gguf) in:  D:\chatbot_RAG\llama.cpp\models\

Place your document files (PDFs and TXT files) in:  D:\chatbot_RAG\data\



**How to Run**

To start the chatbot, run:  python chatbot.py


The chatbot will:

Load the DeepSeek model from the specified location.

Retrieve context from your documents stored in the data folder.

Generate a response based on the combined prompt of user input and the retrieved context.


**Data Store**

The RAG pipeline in rag.py handles your document data:

It reads PDFs using PyPDF2 and TXT files directly.

It splits document text into manageable chunks.

It builds a FAISS vector index using embeddings from the all-MiniLM-L6-v2 sentence-transformer.

When a query is made, the system performs a similarity search to extract the most relevant text as context.


**Usage**

-Prepare Your Data:

Ensure your PDFs and TXT files (containing your knowledge base) are placed in the data folder.

-Run the Chatbot:

python chatbot.py

-Interact with the Bot:

For example, if your TXT file contains the text "Ash's favourite colour is sea blue," ask: *What is Ash's favourite colour?*

The chatbot will retrieve the relevant context and generate an answer accordingly.

**Contributing**

Contributions are welcome! If you have suggestions or improvements, please open an issue or submit a pull request.

**License**

This project is licensed under the MIT License. See the LICENSE file for details.

**Commands Summary**

Below is a quick list of the commands you might use in your setup:

Clone the Repository:  git clone https://github.com/SNEAKO7/chatbot_RAG.git

                       cd chatbot_RAG

Set Up Virtual Environment:

Windows (Command Prompt): python -m venv venv

                          venv\Scripts\activate

Install Dependencies:  pip install llama-cpp-python PyPDF2 langchain faiss-cpu sentence-transformers git+https://github.com/langchain-ai/langchain-community.git

Generate Requirements File (Optional): pip freeze > requirements.txt

Set Up Git LFS (if needed):  git lfs install

                             git lfs track "*.gguf"
                             
Create .gitignore File (to exclude unnecessary files):

Create a file named .gitignore with the following content:

venv/

models/

data/

__pycache__/

*.dll

*.lib


-Commit and Push:

git add .

git commit -m "Initial commit"

git push -u origin main


**Downloading the DeepSeek Model & Installing Ollama**

1. Download the DeepSeek Model
   
You can download the deepseek-r1-distill-qwen-7b-q4_k_m.gguf model from Hugging Face:
🔗 DeepSeek Models on Hugging Face - https://huggingface.co/Kondara/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M-GGUF
 place it inside-  D:\chatbot_RAG\llama.cpp\models\

**What is llama.cpp and Why Do We Need It?**

llama.cpp is a lightweight and optimized C++ implementation of LLaMA (Large Language Model Meta AI). It allows you to run LLaMA-based models, including DeepSeek, efficiently on local hardware without needing a powerful GPU.

We need llama.cpp because:

It enables running DeepSeek LLM locally without requiring high-end hardware.

It supports GGUF format models, which are optimized for CPU inference.

It is efficient and provides low-latency text generation for our RAG chatbot.

Since we are using DeepSeek LLM, which is available in GGUF format, llama.cpp is necessary to load and run the model on your machine.

You need to download and place llama.cpp inside your project folder (chatbot_RAG). This ensures that all dependencies are in one place and makes it easier to run the chatbot without configuring paths separately.

**Steps to Download and Set Up llama.cpp:**

Download llama.cpp from the official repository:  git clone https://github.com/ggerganov/llama.cpp.git

Move it inside your project folder (chatbot_RAG): mv llama.cpp chatbot_RAG/

Navigate into the llama.cpp folder and build it:  cd chatbot_RAG/llama.cpp

                                                  make

                                                 


Enjoy building and enhancing your local RAG chatbot with Chatbot_RAG!

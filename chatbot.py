from llama_cpp import Llama
from rag import retrieve_context

# Path to your DeepSeek model (adjust if necessary)
model_path = r"D:\chatbot_RAG\llama.cpp\models\deepseek-r1-distill-qwen-7b-q4_k_m.gguf"
llm = Llama(model_path=model_path)

# Path to the folder containing your PDFs and TXT files
docs_folder = r"D:\chatbot_RAG\data"

# Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    # Retrieve context from the specified folder.
    context = retrieve_context(user_input, docs_folder)

    # Build a prompt that instructs the model to use only the provided context.
    prompt = (
        f"Answer the following question using ONLY the provided context. "
        f"If the context does not mention the answer, say 'Context does not provide enough information.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {user_input}\nAnswer:"
    )

    response = llm(prompt, max_tokens=200)
    answer = response["choices"][0]["text"].strip()
    print("AI:", answer)


'''
#added bye and greetings

from llama_cpp import Llama
from rag import retrieve_context
import os

# Ensure model path exists
model_path = r"D:\chatbot_RAG\llama.cpp\models\mistral-7b-instruct-v0.1-q4_k_m.gguf"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

# Load Llama model with correct context window
llm = Llama(model_path=model_path, n_ctx=8192)  

# Path to the folder containing your PDFs and TXT files
docs_folder = r"D:\chatbot_RAG\data"

# Greeting message
print("\nWelcome! I'm your AI assistant. Ask me anything, and I'll do my best to assist you.")
print("If you want to exit, just type 'exit', 'quit', or 'bye'.\n")

# Chat loop
while True:
    user_input = input("You: ").strip()
    
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("AI: Goodbye! Have a great day!\n")
        break

    # Retrieve context from documents
    context = retrieve_context(user_input, docs_folder)

    # Build a proper prompt to guide the model
    prompt = (
        f"Use the given context to answer the question accurately. "
        f"If the context does not provide enough information, say 'Context does not provide enough details.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {user_input}\nAnswer:"
    )

    try:
        response = llm(prompt, max_tokens=3000)  # Limit tokens to prevent overflow
        answer = response["choices"][0]["text"].strip()
        print("AI:", answer)
        print("\nFeel free to ask more questions! If it's in my database, I'll surely help you out.\n")
    except Exception as e:
        print("Error:", e)

        
#REMOVING UNESECARY LOGS IN OUTPUT


import warnings
from llama_cpp import Llama
from rag import retrieve_context
import os

# Suppress warnings from LangChain
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure model path exists
model_path = r"D:\chatbot_RAG\llama.cpp\models\mistral-7b-instruct-v0.1-q4_k_m.gguf"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

# Load Llama model with correct context window and suppress logs
llm = Llama(model_path=model_path, n_ctx=8192, verbose=False)

# Path to the folder containing your PDFs and TXT files
docs_folder = r"D:\chatbot_RAG\data"

# Greeting message
print("AI: Hello! Ask me anything, and I'll try my best to help you.")

# Chat loop
while True:
    user_input = input("You: ").strip()
    
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("AI: Goodbye! Have a great day!")
        break

    # Retrieve context from documents
    context = retrieve_context(user_input, docs_folder)

    # Build a proper prompt to guide the model
    prompt = (
        f"Use the given context to answer the question accurately. "
        f"If the context does not provide enough information, say 'Context does not provide enough details.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {user_input}\nAnswer:"
    )

    try:
        response = llm(prompt, max_tokens=3000)  # Limit tokens to prevent overflow
        answer = response["choices"][0]["text"].strip()
        print(f"AI: {answer}\n")
        print("AI: Feel free to ask more questions! If it's in my database, I'll surely help you out.")
    except Exception as e:
        print("AI: Error processing your request.")'''
''' #UPDATED WITH CAL
import warnings
import os
from llama_cpp import Llama
from rag import retrieve_context

# Suppress warnings from LangChain
warnings.filterwarnings("ignore", category=UserWarning)

# Model path
model_path = r"D:\chatbot_RAG\llama.cpp\models\mistral-7b-instruct-v0.1-q4_k_m.gguf"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

# Load Llama model
llm = Llama(model_path=model_path, n_ctx=8192, verbose=False)

# Document folder path
docs_folder = r"D:\chatbot_RAG\data"

# Greeting message
print("\nCal: Hi! I'm Cal, the Callippus assistant. I help retrieve information from FRS and UAT documents.\n")

# Chat history storage (Limited to last 5 interactions for context)
chat_history = []

while True:
    user_input = input("You: ").strip()
    
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Cal: Goodbye! If you need help again, just ask. Take care! 😊")
        break

    # Retrieve relevant context from documents
    context = retrieve_context(user_input, docs_folder)

    # Constructing prompt for the model
    prompt = f"""You are Cal, the Callippus assistant. Use the provided context to answer accurately.
If context lacks details, say 'The context does not provide enough details.'

Context: {context}

User: {user_input}
Cal:"""

    try:
        response = llm(prompt, max_tokens=500)
        answer = response["choices"][0]["text"].strip()

        if not answer:
            answer = "The context does not provide enough details."

        print(f"Cal: {answer}\n")

        # Maintain chat history (limit to last 5 interactions)
        chat_history.append(f"You: {user_input}\nCal: {answer}")
        if len(chat_history) > 5:
            chat_history.pop(0)

    except Exception as e:
        print("Cal: Sorry, I encountered an error processing your request.")'''

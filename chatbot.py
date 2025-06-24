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
        print("Cal: Sorry, I encountered an error processing your request.")

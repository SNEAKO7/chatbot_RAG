from flask import Flask, render_template, request, jsonify
from rag import retrieve_context
from llama_cpp import Llama
import os

# Initialize Flask app
app = Flask(__name__)

# Model path
model_path = r"D:\chatbot_RAG\llama.cpp\models\mistral-7b-instruct-v0.1-q4_k_m.gguf"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

# Load Llama model
llm = Llama(model_path=model_path, n_ctx=8192, verbose=False)

# Document folder path
docs_folder = r"D:\chatbot_RAG\data"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip()
    
    if not user_input:
        return jsonify({"response": "Please enter a message."})

    # Retrieve context from documents
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

        return jsonify({"response": answer})

    except Exception as e:
        return jsonify({"response": "Sorry, I encountered an error processing your request."})

if __name__ == "__main__":
    app.run(debug=True)

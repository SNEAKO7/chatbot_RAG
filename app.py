import os
from flask import Flask, render_template, request, jsonify
from rag import retrieve_context
from llama_cpp import Llama

app = Flask(__name__)

model_path = "D:/chatbot_RAG/llama.cpp/models/mistral-7b-instruct-v0.1-q4_k_m.gguf"
#model_path = "/app/model/mistral-7b-instruct-v0.1-q4_k_m.gguf"      #For docker
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Llama model file not found: {model_path}")
llm = Llama(model_path=model_path, n_ctx=8192, n_batch=512, verbose=False)

docs_folder = "D:/chatbot_RAG/data"
#docs_folder = "/app/data"
chat_history = []

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"response": "Please enter a message."})

    chat_history.append({"role": "user", "content": user_input})

    context_text, context_sources = retrieve_context(user_input, docs_folder)
    history_text = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in chat_history[-4:]
    )

    if context_text:
        prompt = (
            f"You are Cal, the Callippus assistant. Use the following context to answer the user's question.\n\n"
            f"Conversation history:\n{history_text}\n\n"
            f"Context:\n{context_text}\n\nAnswer:"
        )
    else:
        prompt = (
            f"You are Cal, the Callippus assistant. No matching context was found.\n\n"
            f"Conversation history:\n{history_text}\n\nAnswer:"
        )

    res = llm(prompt, temperature=0, max_tokens=300)
    answer = res["choices"][0]["text"].strip()

    if context_sources:
        sources = sorted(set(src for _, src in context_sources))
        source_note = "\n\n📄 *Context found in:*\n" + "\n".join(f"- {src}" for src in sources)
        answer += source_note

    chat_history.append({"role": "assistant", "content": answer})
    return jsonify({"response": answer})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)


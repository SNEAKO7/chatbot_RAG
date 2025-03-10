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

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from llama_cpp import Llama
import textwrap
import html

def display_results(results):
    print("\n📰 Top Relevant News Articles:\n")
    for i, article in enumerate(results, 1):
        # Clean HTML entities
        clean_text = html.unescape(article)

        # Remove backslashes
        clean_text = clean_text.replace("\\n", " ").replace("\\t", " ").replace("\\", "")

        # Wrap lines for terminal
        wrapped = textwrap.fill(clean_text, width=100)

        print(f"{i}. {wrapped}\n")

# --- 1. Load AG News subset
dataset = load_dataset("ag_news", split="train[:2000]")
texts = [item["text"] for item in dataset]

# --- 2. Setup Chroma vector DB with SentenceTransformer embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

client = chromadb.Client()
collection = client.create_collection(name="agnews", embedding_function=embedding_fn)

# Add documents
collection.add(
    documents=texts,
    ids=[f"id_{i}" for i in range(len(texts))]
)

# --- 3. Accept user query
query = input("Ask a question: ")
results = collection.query(query_texts=[query], n_results=3)

relevant_docs = results['documents'][0]
context = "\n\n".join(relevant_docs)

llm = Llama(model_path="./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf", n_ctx=2048, n_threads=8)

try:
    print("Model loaded successfully!")
    # Your code to interact with the model goes here
    # For example: response = llm.query("Your input query here")
    print("Model loaded successfully!")
    # Your code for querying the model goes here
    prompt = f"""You are a helpful news assistant.

    Based on the following news articles, answer the question in a clear, conversational tone.

    Context:
    {context}

    Question: {query}
    Answer:"""

    response = llm(prompt, max_tokens=300, stop=["</s>"])
    print(response["choices"][0]["text"])


finally:
    # Ensure model cleanup (deallocation) when done
    llm.close()
    print("Model cleaned up successfully.")

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import numpy as np

load_dotenv()

# Step 1: Create the embedder
embedder = OpenAIEmbeddings(model="text-embedding-3-small")

# Step 2: Embed some sentences
sentences = [
    "Demand forecasting at Amazon",
    "Predicting future sales volumes",
    "Neural networks in deep learning",
    "I enjoy cooking pasta",
    "Supply chain optimization"
]

vectors = embedder.embed_documents(sentences)

print(f"Number of sentences: {len(vectors)}")
print(f"Each vector has {len(vectors[0])} numbers")
print(f"First 5 numbers of sentence 1: {vectors[0][:5]}")



# Step 3: Similarity search manually
# Cosine similarity measures how "close" two vectors are
# 1.0 = identical meaning, 0.0 = completely different

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Embed a question
question = "How does Amazon predict demand?"
question_vector = embedder.embed_query(question)

print(f"\nQuestion: '{question}'")
print("\nSimilarity to each sentence:")
for i, sentence in enumerate(sentences):
    score = cosine_similarity(question_vector, vectors[i])
    print(f"  {score:.3f} — {sentence}")


# Step 4: Store vectors in FAISS for fast search
from langchain_community.vectorstores import FAISS

print("\n--- Building FAISS Vector Store ---")

# FAISS stores all vectors and lets you search them fast
vectorstore = FAISS.from_texts(sentences, embedder)

# Now search it with a question
results = vectorstore.similarity_search(
    "How does Amazon predict demand?", 
    k=2  # return top 2 results
)

print("Top 2 most relevant sentences:")
for r in results:
    print(f"  → {r.page_content}")

# Save it to disk so we can reuse it
vectorstore.save_local("day2_vectorstore")
print("\nVector store saved to disk ✅")
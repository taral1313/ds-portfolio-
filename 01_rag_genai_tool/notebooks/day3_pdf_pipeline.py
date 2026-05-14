from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# ── STEP 1: Load the PDF ──────────────────────────────
print("Step 1: Loading PDF...")
loader = PyPDFLoader("data/cv.pdf")
pages = loader.load()

print(f"  Pages loaded: {len(pages)}")
print(f"  First 300 chars of page 1:")
print(f"  {pages[0].page_content[:300]}")

# ── STEP 2: Chunk it ─────────────────────────────────
print("\nStep 2: Chunking...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,    # each chunk ~500 characters
    chunk_overlap=50   # 50 char overlap between chunks
)
chunks = splitter.split_documents(pages)

print(f"  Total chunks created: {len(chunks)}")
print(f"\n  Example chunk:")
print(f"  '{chunks[2].page_content}'")

# ── STEP 3: Embed + Store in FAISS ───────────────────
print("\nStep 3: Embedding chunks and building vector store...")
embedder = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(chunks, embedder)
vectorstore.save_local("cv_vectorstore")
print("  Vector store saved ✅")

# ── STEP 4: Test retrieval ────────────────────────────
print("\nStep 4: Testing retrieval...")
questions = [
    "What ML tools does Taral know?",
    "Where did Taral work?",
    "What is Taral's educational background?"
]

for q in questions:
    print(f"\n  Q: {q}")
    results = vectorstore.similarity_search(q, k=2)
    for r in results:
        print(f"  → {r.page_content[:150]}...")
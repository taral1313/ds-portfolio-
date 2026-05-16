from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

# ── STEP 1: Load ALL PDFs from data folder ────────────
print("Loading documents...")
all_documents = []
pdf_folder = "data/"

for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        print(f"  Loading: {filename}")
        loader = PyPDFLoader(os.path.join(pdf_folder, filename))
        docs = loader.load()
        
        # Tag each chunk with its source filename
        for doc in docs:
            doc.metadata["source"] = filename
        
        all_documents.extend(docs)

print(f"\nTotal pages loaded: {len(all_documents)}")

# ── STEP 2: Chunk with larger size for financial docs ─
# Financial reports have dense content, 
# slightly larger chunks preserve more context
splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100
)
chunks = splitter.split_documents(all_documents)
print(f"Total chunks: {len(chunks)}")

# ── STEP 3: Embed + Store ─────────────────────────────
print("\nBuilding vector store (this takes ~30 seconds)...")
embedder = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(chunks, embedder)
vectorstore.save_local("financial_vectorstore")
print("Saved ✅")

# ── STEP 4: Build RAG chain ───────────────────────────
prompt_template = """
You are a financial analyst assistant. Answer questions 
about company financial reports accurately and concisely.
Always mention which company you are referring to.
If the information isn't in the context, say so clearly.

Context:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

# ── STEP 5: Ask financial questions ───────────────────
questions = [
    "What was the total revenue last year?",
    "What were the main risks mentioned?",
    "How many employees does the company have?",
    "What are the key business segments?"
]

print("\n" + "="*60)
for q in questions:
    print(f"\n❓ {q}")
    result = qa_chain.invoke({"query": q})
    print(f"💬 {result['result']}")
    
    # Show which document the answer came from
    sources = set(d.metadata.get('source', 'unknown') 
                  for d in result['source_documents'])
    print(f"📄 Source: {sources}")
    print("-"*60)
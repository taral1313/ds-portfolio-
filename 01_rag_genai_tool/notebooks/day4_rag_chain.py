from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

# ── STEP 1: Load your saved vectorstore ──────────────
# No API call needed - loading from disk is instant + free
print("Loading vector store from disk...")
embedder = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.load_local(
    "cv_vectorstore",
    embedder,
    allow_dangerous_deserialization=True
)
print("Loaded ✅")

# ── STEP 2: Define a prompt template ─────────────────
# This controls HOW GPT uses the retrieved chunks
# {context} = the chunks FAISS retrieved
# {question} = the user's question
prompt_template = """
You are a helpful assistant answering questions about 
a candidate's CV and background.

Use ONLY the context below to answer. 
If the answer isn't in the context, say 
"I don't have that information in the CV."

Context:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# ── STEP 3: Build the RAG chain ───────────────────────
# This connects: question → FAISS retrieval → GPT → answer
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",        # "stuff" = put all chunks into one prompt
    retriever=vectorstore.as_retriever(
        search_kwargs={"k": 3} # retrieve top 3 chunks
    ),
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True  # show which chunks were used
)

# ── STEP 4: Ask questions and see the magic ───────────
questions = [
    "What was Taral's biggest achievement at Amazon?",
    "What programming languages does Taral know?",
    "What is Taral's educational background?",
    "Does Taral have experience with deep learning?",
    "What is Taral's favourite food?"  # not in CV - tests hallucination guard
]

print("\n" + "="*60)
for q in questions:
    print(f"\n❓ Question: {q}")
    result = qa_chain.invoke({"query": q})
    print(f"💬 Answer: {result['result']}")
    print(f"📄 Sources used: {len(result['source_documents'])} chunks")
    print("-"*60)
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

# ── PAGE CONFIG ───────────────────────────────────────
st.set_page_config(
    page_title="Financial Report Analyser",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Financial Report Analyser")
st.markdown("Ask anything about the uploaded financial reports.")
st.divider()

# ── LOAD VECTORSTORE ──────────────────────────────────
@st.cache_resource
def load_vectorstore():
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")

    if os.path.exists("financial_vectorstore"):
        return FAISS.load_local(
            "financial_vectorstore",
            embedder,
            allow_dangerous_deserialization=True
        )

    # Build from PDFs if vectorstore doesn't exist
    st.info("Building vector store from documents...")
    all_documents = []
    for filename in os.listdir("data/"):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join("data/", filename))
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = filename
            all_documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(all_documents)
    vectorstore = FAISS.from_documents(chunks, embedder)
    vectorstore.save_local("financial_vectorstore")
    return vectorstore

# ── BUILD RAG CHAIN ───────────────────────────────────
@st.cache_resource
def load_chain():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Prompt that includes chat history for memory
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a financial analyst assistant.
Answer questions about company financial reports accurately.
Always mention which company you are referring to.
If the answer isn't in the context, say so clearly.
Do not make up numbers or facts.

Context:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, chain)

# ── INITIALISE SESSION STATE ──────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── SIDEBAR ───────────────────────────────────────────
with st.sidebar:
    st.header("📁 Loaded Reports")
    if os.path.exists("data/"):
        for f in os.listdir("data/"):
            if f.endswith(".pdf"):
                st.markdown(f"- {f}")
    st.divider()
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()
    st.divider()
    st.caption("Built by Taral Sarvagod")
    st.caption("Stack: LangChain · OpenAI · FAISS · Streamlit")

# ── LOAD CHAIN ────────────────────────────────────────
with st.spinner("Loading financial reports..."):
    chain = load_chain()
st.success("Ready! Ask me anything about the reports.")

# ── DISPLAY CHAT HISTORY ──────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── CHAT INPUT ────────────────────────────────────────
if question := st.chat_input("Ask about the financial reports..."):

    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("assistant"):
        with st.spinner("Analysing reports..."):
            result = chain.invoke({
                "input": question,
                "chat_history": st.session_state.chat_history
            })
            answer = result["answer"]
            sources = set(
                doc.metadata.get("source", "unknown")
                for doc in result["context"]
            )

        st.markdown(answer)
        with st.expander("📄 Sources"):
            for s in sources:
                st.markdown(f"- `{s}`")

    # Update chat history for memory
    st.session_state.chat_history.extend([
        HumanMessage(content=question),
        AIMessage(content=answer)
    ])
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
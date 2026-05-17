import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# ── ABSOLUTE PATHS ─────────────────────────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
VECTORSTORE_DIR = os.path.join(APP_DIR, "financial_vectorstore")

# ── PAGE CONFIG ────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Report Analyser",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Financial Report Analyser")
st.markdown("Ask anything about the uploaded financial reports.")
st.divider()

# ── LOAD VECTORSTORE ───────────────────────────────────────
@st.cache_resource
def load_vectorstore():
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")

    if os.path.exists(VECTORSTORE_DIR):
        return FAISS.load_local(
            VECTORSTORE_DIR,
            embedder,
            allow_dangerous_deserialization=True
        )

    st.info("Building vector store from documents...")
    all_documents = []

    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_DIR, filename))
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
    vectorstore.save_local(VECTORSTORE_DIR)
    return vectorstore

# ── LOAD RETRIEVER ─────────────────────────────────────────
@st.cache_resource
def load_retriever():
    vectorstore = load_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": 4})

# ── INITIALISE SESSION STATE ───────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── SIDEBAR ────────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Loaded Reports")
    if os.path.exists(DATA_DIR):
        for f in os.listdir(DATA_DIR):
            if f.endswith(".pdf"):
                st.markdown(f"- 📄 {f}")
    st.divider()
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()
    st.divider()
    st.caption("Built by Taral Sarvagod")
    st.caption("Stack: LangChain · OpenAI · FAISS · Streamlit")

# ── LOAD ───────────────────────────────────────────────────
with st.spinner("Loading financial reports..."):
    retriever = load_retriever()
st.success("Ready! Ask me anything about the reports.")

# ── DISPLAY CHAT HISTORY ───────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── CHAT INPUT ─────────────────────────────────────────────
if question := st.chat_input("Ask about the financial reports..."):

    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("assistant"):
        with st.spinner("Analysing reports..."):

            # Retrieve relevant chunks
            source_docs = retriever.invoke(question)
            context_text = "\n\n".join(
                doc.page_content for doc in source_docs
            )
            sources = set(
                doc.metadata.get("source", "unknown")
                for doc in source_docs
            )

            # Build prompt and call LLM directly
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a financial analyst assistant.
Answer questions about company financial reports accurately.
Always mention which company you are referring to.
If the answer is not in the context below, say clearly
that you don't have that information. Never make up numbers.

Context: {context}"""),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}")
            ])

            messages = prompt.format_messages(
                context=context_text,
                chat_history=st.session_state.chat_history,
                question=question
            )
            response = llm.invoke(messages)
            answer = response.content

        st.markdown(answer)
        with st.expander("📄 Sources"):
            for s in sources:
                st.markdown(f"- `{s}`")

    st.session_state.chat_history.extend([
        HumanMessage(content=question),
        AIMessage(content=answer)
    ])
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
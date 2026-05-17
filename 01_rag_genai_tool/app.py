import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

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

# ── BUILD RAG CHAIN (pure langchain_core, no langchain.chains) ───
@st.cache_resource
def load_chain():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
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

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Pure LCEL chain — no langchain.chains imports needed
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "chat_history": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever

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
                st.markdown(f"- 📄 {f}")
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
    chain, retriever = load_chain()
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

            # Get source documents separately for citation
            source_docs = retriever.invoke(question)
            sources = set(
                doc.metadata.get("source", "unknown")
                for doc in source_docs
            )

            # Run the chain
            answer = chain.invoke({
                "question": question,
                "chat_history": st.session_state.chat_history,
                "context": "\n\n".join(
                    doc.page_content for doc in source_docs
                )
            })

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
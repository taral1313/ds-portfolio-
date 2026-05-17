import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

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

# ── LOAD VECTORSTORE + BUILD CHAIN ────────────────────
@st.cache_resource
def load_vectorstore():
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(
        "financial_vectorstore",
        embedder,
        allow_dangerous_deserialization=True
    )
    return vectorstore

vectorstore = load_vectorstore()

# ── MEMORY: lives in session_state so it persists ─────
# Each user session gets its own memory object
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── BUILD CHAIN WITH MEMORY ───────────────────────────
# ConversationalRetrievalChain automatically includes
# chat history in every prompt it sends to GPT
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    memory=st.session_state.memory,
    return_source_documents=True,
    verbose=False
)

# ── SIDEBAR ───────────────────────────────────────────
with st.sidebar:
    st.header("📁 Loaded Reports")
    st.markdown("- Amazon Annual Report 2023")
    st.markdown("- *(add more PDFs to data/ folder)*")
    st.divider()

    # Clear conversation button
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.rerun()

    st.divider()
    st.caption("Built by Taral Sarvagod")
    st.caption("Stack: LangChain · OpenAI · FAISS · Streamlit")

# ── DISPLAY CHAT HISTORY ──────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── CHAT INPUT ────────────────────────────────────────
if question := st.chat_input("Ask about the financial reports..."):

    # Show user question immediately
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # Get answer with memory
    with st.chat_message("assistant"):
        with st.spinner("Analysing reports..."):
            result = qa_chain.invoke({"question": question})
            answer = result["answer"]
            sources = set(
                doc.metadata.get("source", "unknown")
                for doc in result["source_documents"]
            )

        st.markdown(answer)

        with st.expander("📄 Sources"):
            for s in sources:
                st.markdown(f"- `{s}`")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
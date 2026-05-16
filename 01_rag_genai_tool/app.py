import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

load_dotenv()

# ── PAGE CONFIG ───────────────────────────────────────
st.set_page_config(
    page_title="Financial Report Analyser",
    page_icon="📊",
    layout="centered"
)

# ── HEADER ────────────────────────────────────────────
st.title("📊 Financial Report Analyser")
st.markdown("Ask any question about the uploaded financial reports.")
st.divider()

# ── LOAD VECTORSTORE (cached so it only loads once) ───
@st.cache_resource
def load_qa_chain():
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(
        "financial_vectorstore",
        embedder,
        allow_dangerous_deserialization=True
    )

    prompt_template = """
You are a financial analyst assistant. Answer questions 
about company financial reports accurately and concisely.
Always mention which company you are referring to.
If the information isn't in the context, say so clearly.
Do not make up numbers or facts.

Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return chain

# ── LOAD THE CHAIN ────────────────────────────────────
with st.spinner("Loading financial reports..."):
    qa_chain = load_qa_chain()
st.success("Ready! Ask me anything about the reports.")

# ── CHAT HISTORY ──────────────────────────────────────
# Store conversation in session so it persists while app runs
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── CHAT INPUT ────────────────────────────────────────
if question := st.chat_input("Ask a question about the financial reports..."):

    # Show user question
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Searching reports..."):
            result = qa_chain.invoke({"query": question})
            answer = result["result"]

            # Get source documents
            sources = set(
                doc.metadata.get("source", "unknown")
                for doc in result["source_documents"]
            )

            # Display answer
            st.markdown(answer)

            # Display sources in a nice expander
            with st.expander("📄 Sources used"):
                for source in sources:
                    st.markdown(f"- `{source}`")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
# 📊 Financial Report Analyser — RAG-based GenAI Tool

A conversational AI tool that lets you ask natural language questions 
to financial reports and get accurate, cited answers in seconds.

🔗 **[Live Demo →](https://financial-analyser-tool.streamlit.app/)**

---

## 🎯 What It Does

Upload any financial PDF and ask questions like:
- *"What was Amazon's net income in 2023?"*
- *"What are the main business risks mentioned?"*
- *"How did Meta's revenue compare across years?"*

The tool finds the exact relevant sections and answers with source citations — 
no hallucination, no made-up numbers.

---

## 🏗️ Architecture
User Question
│
▼
OpenAI Embeddings          ← converts question to vector
│
▼
FAISS Vector Store         ← finds top 4 most relevant chunks
│
▼
GPT-3.5-turbo + Prompt     ← generates answer from retrieved context
│
▼
Streamlit UI               ← displays answer + source citation

This is **Retrieval-Augmented Generation (RAG)** — combining semantic 
search with LLM generation for accurate, grounded answers.

---

## 🧠 Key Concepts Implemented

| Concept | Implementation |
|---|---|
| Text Embeddings | OpenAI text-embedding-3-small (1536 dimensions) |
| Vector Store | FAISS for fast similarity search |
| Chunking Strategy | 700 chars, 100 char overlap |
| Hallucination Guard | Prompt instructs model to only use provided context |
| Conversation Memory | Full chat history passed with each query |
| Source Citation | Every answer shows which document it came from |

---

## 📂 Project Structure
01_rag_genai_tool/
├── app.py                      ← Streamlit web app
├── notebooks/
│   ├── day2_embeddings.py      ← embedding exploration
│   ├── day3_pdf_pipeline.py    ← PDF ingestion pipeline
│   └── day4_rag_chain.py       ← full RAG chain prototype
├── data/                       ← financial report PDFs
│   ├── Amazon-com-Inc-2023-Annual-Report.pdf
│   ├── meta-annual-report-2023.pdf
│   └── tsla-20231231-gen.pdf
└── requirements.txt

---

## 🛠️ Tech Stack

- **LangChain** — RAG pipeline orchestration
- **OpenAI API** — embeddings + GPT-3.5-turbo
- **FAISS** — vector similarity search
- **Streamlit** — web app and deployment
- **PyPDF** — PDF text extraction

---

## 🚀 Run Locally

```bash
git clone https://github.com/taral1313/ds-portfolio-.git
cd ds-portfolio-/01_rag_genai_tool
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "OPENAI_API_KEY=your-key-here" > .env
streamlit run app.py
```

---

## 💡 What I Learned Building This

- How RAG works end-to-end — from PDF ingestion to LLM response
- Why chunking strategy and overlap size affect retrieval quality
- How to prevent hallucination using prompt engineering
- Difference between local and cloud deployment (path handling,
  secrets management, dependency conflicts)
- How FAISS performs semantic search using cosine similarity

---

## 🔗 Related Projects

- [Project 2 — Demand Forecasting](../02_demand_forecasting/) *(coming soon)*
- [Project 3 — Customer Churn Prediction](../03_customer_churn/) *(coming soon)*

---

*Built by [Taral Sarvagod](https://linkedin.com/in/taral-sarvagod-20ba00190) — Data Scientist | ex-Amazon Luxembourg*

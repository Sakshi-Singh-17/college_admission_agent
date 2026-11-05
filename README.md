# 🎓 College Admission Agent (RAG-Based)

An AI-powered college admissions assistant built with **Retrieval-Augmented Generation (RAG)**.
Students can ask natural language questions and receive accurate, sourced answers about
admissions at Greenfield University — covering policies, eligibility, fees, courses, and deadlines.

---

## 📐 Architecture

```
Student Question
       │
       ▼
┌─────────────────────────────────────────────────────┐
│             Conversational RAG Chain                │
│                                                     │
│  ┌───────────────┐      ┌──────────────────────┐   │
│  │  FAISS Vector │      │  Conversation Memory  │   │
│  │  Store (MMR)  │      │  (Window Buffer k=10) │   │
│  └──────┬────────┘      └──────────┬────────────┘   │
│         └─────────────┬────────────┘                │
│                  ┌────▼─────┐                       │
│                  │   LLM    │  OpenAI / Ollama       │
│                  └────┬─────┘                       │
└───────────────────────┼─────────────────────────────┘
                        │
                ┌───────▼────────┐
                │ Answer + Sources│
                └────────────────┘
```

### Key Components

| File | Role |
|------|------|
| `config.py` | Environment-driven settings (LLM, embeddings, paths) |
| `rag_pipeline.py` | Document loading → chunking → embedding → FAISS index |
| `agent.py` | ConversationalRetrievalChain with memory + custom prompts |
| `main.py` | Rich CLI chat interface (interactive + demo modes) |
| `build_index.py` | Standalone script to (re)build the vector index |
| `test_agent.py` | End-to-end functional test suite |
| `data/knowledge_base/` | Institutional documents (policies, fees, FAQs, …) |
| `data/vector_store/` | Persisted FAISS index (auto-generated) |

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running (for local/offline mode)

```bash
# Install and start Ollama
ollama serve
ollama pull llama3.2
```

### 2. Set up Python environment

```bash
cd college_admission_agent
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure environment

The project ships with a `.env` pre-configured for **local / offline** mode
(Ollama LLM + sentence-transformers embeddings — no API key needed).

To use **OpenAI** instead, edit `.env`:
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_PROVIDER=openai   # optional: uses text-embedding-3-small
```

### 4. Build the knowledge base index

```bash
python build_index.py --stats
```

### 5. Start the admission assistant

```bash
# Interactive chat
python main.py

# Force-rebuild index then chat
python main.py --rebuild

# Non-interactive demo (preset questions)
python main.py --demo
```

---

## 💬 Sample Interaction

```
You: What are the eligibility criteria for B.Tech CSE?

🎓 Admission Assistant:
  To be eligible for B.Tech CSE at Greenfield University, you need:
  • Completion of 10+2 with Physics, Chemistry, and Mathematics (PCM)
  • Minimum aggregate of **60%** (55% for SC/ST/OBC candidates)
  • JEE Main score — minimum **75th percentile** required
  ...

📄 Sources Consulted:
  • Admission Policies
  • Eligibility Criteria
  • Courses Catalogue
```

### Available CLI commands
| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/reset` | Clear conversation history |
| `/history` | View full conversation |
| `/sources` | Toggle source document display |
| `/config` | Show active configuration |
| `/quit` | Exit the application |

---

## 📚 Knowledge Base

Five institutional documents are included out of the box:

| Document | Contents |
|----------|----------|
| `admission_policies.txt` | UG/PG/Ph.D. policies, reservations, cancellation rules |
| `courses_catalogue.txt` | All programmes, intake, eligibility, specialisations |
| `fee_structure.txt` | Tuition, hostel, scholarships, payment schedule |
| `deadlines_calendar.txt` | Application dates, exam schedule, academic calendar |
| `eligibility_criteria.txt` | Stream-wise & category-wise eligibility table |
| `faqs.txt` | 23 FAQs covering application, fees, hostel, contact info |

### Adding new documents

1. Add `.txt` (or `.pdf`) files to `data/knowledge_base/`
2. Re-run `python build_index.py --force`

---

## 🧪 Running Tests

```bash
python test_agent.py
```

Tests cover:
1. Knowledge base file existence
2. Document loading
3. Text splitting / chunk sizing
4. Vector store build
5. Semantic retrieval correctness
6. Agent end-to-end response quality
7. Conversational memory and reset

---

## ⚙️ Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | `openai` or `ollama` |
| `OPENAI_API_KEY` | *(empty)* | Required if `LLM_PROVIDER=openai` |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `EMBEDDING_PROVIDER` | `local` | `local` (MiniLM) or `openai` |
| `TOP_K_RESULTS` | `5` | Chunks retrieved per query |
| `MAX_HISTORY_TURNS` | `10` | Conversation turns kept in memory |
| `CHUNK_SIZE` | `800` | Max characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between adjacent chunks |

---

## 🗂 Project Structure

```
college_admission_agent/
├── .env                        # Active config (pre-set for local mode)
├── .env.example                # Commented template
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── config.py                   # Settings loader
├── rag_pipeline.py             # RAG: load → chunk → embed → FAISS
├── agent.py                    # Conversational agent
├── main.py                     # CLI interface
├── build_index.py              # Index builder utility
├── test_agent.py               # Functional test suite
│
└── data/
    ├── knowledge_base/         # Source documents (editable)
    │   ├── admission_policies.txt
    │   ├── courses_catalogue.txt
    │   ├── deadlines_calendar.txt
    │   ├── eligibility_criteria.txt
    │   ├── faqs.txt
    │   └── fee_structure.txt
    └── vector_store/           # Auto-generated FAISS index
        ├── index.faiss
        └── index.pkl
```

---

## 🔧 Extending the Agent

**Add a new knowledge domain** — drop a `.txt` file in `data/knowledge_base/` and rebuild the index.

**Swap the LLM** — change `LLM_PROVIDER` + model name in `.env`. No code changes needed.

**Add a web UI** — import `AdmissionAgent` from `agent.py` into a FastAPI or Streamlit app.

**Use a different vector store** — replace the FAISS calls in `rag_pipeline.py` with Chroma, Pinecone, or Weaviate — the rest of the pipeline is unchanged.

---

## 📄 License

MIT License — free to use, modify, and distribute.

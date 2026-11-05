"""
agent.py — The College Admission Agent powered by RAG + conversational memory.

Architecture:
  ┌──────────────────┐
  │   Student Query  │
  └────────┬─────────┘
           │
  ┌────────▼─────────────────────────────────┐
  │  Conversational RAG Chain (LCEL)         │
  │  ┌─────────────┐   ┌──────────────────┐  │
  │  │  Retriever  │   │  Chat History    │  │
  │  │  (FAISS MMR)│   │  (in-memory list)│  │
  │  └──────┬──────┘   └────────┬─────────┘  │
  │         └──────────┬────────┘            │
  │              ┌─────▼──────┐              │
  │              │    LLM     │              │
  │              │(OpenAI/Ollama)            │
  │              └─────┬──────┘              │
  └────────────────────┼────────────────────┘
                       │
               ┌───────▼────────┐
               │  Agent Answer  │
               │ + Source Docs  │
               └────────────────┘
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from config import settings
from rag_pipeline import RAGPipeline


# ── Prompt templates ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert College Admission Assistant for Greenfield University.
Your role is to help prospective students with accurate, helpful, and friendly answers about:
  • Admission policies and procedures
  • Eligibility criteria for various programmes
  • Course and programme selection guidance
  • Fee structure and scholarship information
  • Important dates and application deadlines
  • Campus facilities and student life

INSTRUCTIONS:
1. Answer ONLY based on the provided context from the university's official documents.
2. If the context does not contain the answer, say "I don't have that specific information \
in my knowledge base. Please contact admissions@greenfielduniversity.edu.in or \
call +91-20-2700-1234 for accurate details."
3. Always be polite, encouraging, and student-friendly.
4. When discussing fees, always mention scholarship options if relevant.
5. When discussing eligibility, mention reservation/relaxation norms if applicable.
6. For deadlines, always remind students to confirm on the official portal.
7. Structure your answers clearly — use bullet points or numbered lists when listing \
multiple items.

CONTEXT FROM OFFICIAL DOCUMENTS:
{context}"""


# ── LLM factory ──────────────────────────────────────────────────────────────

def _get_llm():
    """Instantiate and return the configured LLM."""
    if settings.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0.3,
            openai_api_key=settings.OPENAI_API_KEY,
        )
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0.3,
        )


# ── Context formatting ────────────────────────────────────────────────────────

def _format_docs(docs: List[Document]) -> str:
    """Concatenate retrieved document chunks into a single context string."""
    parts = []
    for doc in docs:
        src = doc.metadata.get("source", "Document")
        parts.append(f"[Source: {src}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# ── Standalone question rewriter ──────────────────────────────────────────────

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given the conversation history and a follow-up question, rewrite the "
     "follow-up question as a fully standalone question that captures all "
     "necessary context. Output ONLY the standalone question, nothing else."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


# ── Agent ─────────────────────────────────────────────────────────────────────

class AdmissionAgent:
    """
    Conversational RAG agent for college admissions.

    Call `ask(question)` to get a response dict:
      {
        "answer": str,
        "sources": List[str],   # document names used
        "turn_count": int,
      }
    """

    def __init__(self, pipeline: RAGPipeline) -> None:
        self._pipeline = pipeline
        self._llm = _get_llm()
        self._chat_history: List[BaseMessage] = []
        self._retriever = pipeline.get_retriever()
        self._last_source_docs: List[Document] = []

        # QA prompt — chat_history is a list of BaseMessage objects here
        self._qa_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])

    # ── internal helpers ──────────────────────────────────────────────────────

    def _condense_question(self, question: str) -> str:
        """
        Rewrite the user's question as a standalone query when history exists.
        Falls back to the raw question if history is empty.
        """
        if not self._chat_history:
            return question
        condense_chain = CONDENSE_PROMPT | self._llm | StrOutputParser()
        return condense_chain.invoke({
            "chat_history": self._chat_history,
            "question": question,
        })

    def _retrieve(self, standalone_question: str) -> List[Document]:
        docs = self._retriever.invoke(standalone_question)
        self._last_source_docs = docs
        return docs

    def _answer(self, question: str, context: str) -> str:
        chain = self._qa_prompt | self._llm | StrOutputParser()
        return chain.invoke({
            "context": context,
            "chat_history": self._chat_history,
            "question": question,
        })

    # ── public API ────────────────────────────────────────────────────────────

    def ask(self, question: str) -> Dict:
        """
        Submit a question and return the agent's response.

        Returns:
            dict with keys: answer, sources, turn_count
        """
        # 1. Rewrite for retrieval
        standalone = self._condense_question(question)

        # 2. Retrieve context
        docs = self._retrieve(standalone)
        context = _format_docs(docs)

        # 3. Generate answer
        answer = self._answer(question, context)

        # 4. Update history
        self._chat_history.append(HumanMessage(content=question))
        self._chat_history.append(AIMessage(content=answer))

        # Keep only the last MAX_HISTORY_TURNS * 2 messages
        max_msgs = settings.MAX_HISTORY_TURNS * 2
        if len(self._chat_history) > max_msgs:
            self._chat_history = self._chat_history[-max_msgs:]

        # Deduplicate sources
        sources = list({
            doc.metadata.get("source", "Unknown Document")
            for doc in docs
        })

        return {
            "answer": answer,
            "sources": sources,
            "turn_count": len(self._chat_history) // 2,
        }

    def reset_memory(self) -> None:
        """Clear conversation history (start fresh session)."""
        self._chat_history.clear()
        self._last_source_docs = []

    def get_history(self) -> List[Tuple[str, str]]:
        """Return list of (human, ai) message pairs from memory."""
        pairs = []
        messages = self._chat_history
        for i in range(0, len(messages) - 1, 2):
            h = messages[i].content if isinstance(messages[i], HumanMessage) else ""
            a = messages[i + 1].content if isinstance(messages[i + 1], AIMessage) else ""
            pairs.append((h, a))
        return pairs

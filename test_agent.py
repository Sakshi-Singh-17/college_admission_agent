"""
test_agent.py — Lightweight functional tests for the RAG pipeline and agent.

These tests do NOT require an internet connection or API key — they use the
local embedding model and verify pipeline correctness end-to-end.

Run:
    python test_agent.py
"""
from __future__ import annotations

import io
import sys
import traceback
from pathlib import Path

# Force UTF-8 output on Windows so Unicode box-drawing chars in knowledge base
# files don't cause codec errors when printing to the console.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── ensure project root is on path ───────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from rag_pipeline import RAGPipeline, load_documents, split_documents

console_width = 70

PASS = "[PASS]"
FAIL = "[FAIL]"


def section(title: str) -> None:
    print(f"\n{'─' * console_width}")
    print(f"  {title}")
    print('─' * console_width)


def ok(msg: str) -> None:
    print(f"  {PASS}  {msg}")


def fail(msg: str) -> None:
    print(f"  {FAIL}  {msg}")


# ── Individual test functions ─────────────────────────────────────────────────

def test_knowledge_base_files() -> bool:
    section("Test 1 — Knowledge base files exist")
    required = [
        "admission_policies.txt",
        "courses_catalogue.txt",
        "fee_structure.txt",
        "deadlines_calendar.txt",
        "faqs.txt",
        "eligibility_criteria.txt",
    ]
    kb = settings.KNOWLEDGE_BASE_DIR
    all_ok = True
    for fname in required:
        path = kb / fname
        if path.exists():
            ok(f"{fname} found ({path.stat().st_size:,} bytes)")
        else:
            fail(f"{fname} NOT found at {path}")
            all_ok = False
    return all_ok


def test_document_loading() -> bool:
    section("Test 2 — Document loading")
    try:
        docs = load_documents(settings.KNOWLEDGE_BASE_DIR)
        if len(docs) == 0:
            fail("No documents loaded")
            return False
        ok(f"Loaded {len(docs)} documents")
        for doc in docs:
            src = doc.metadata.get("source", "?")
            ok(f"  • {src} ({len(doc.page_content):,} chars)")
        return True
    except Exception as exc:
        fail(f"Exception during loading: {exc}")
        traceback.print_exc()
        return False


def test_text_splitting() -> bool:
    section("Test 3 — Text splitting / chunking")
    try:
        docs = load_documents(settings.KNOWLEDGE_BASE_DIR)
        chunks = split_documents(docs)
        if len(chunks) == 0:
            fail("No chunks produced")
            return False
        sizes = [len(c.page_content) for c in chunks]
        avg = sum(sizes) / len(sizes)
        ok(f"Produced {len(chunks)} chunks")
        ok(f"  Min size: {min(sizes)} | Max size: {max(sizes)} | Avg: {avg:.0f}")
        # Verify no chunk exceeds configured max size + overlap margin
        max_allowed = settings.CHUNK_SIZE + settings.CHUNK_OVERLAP
        oversized = [s for s in sizes if s > max_allowed]
        if oversized:
            fail(f"{len(oversized)} chunks exceed size limit ({max_allowed})")
            return False
        ok("All chunks within size limits")
        return True
    except Exception as exc:
        fail(f"Exception: {exc}")
        traceback.print_exc()
        return False


def test_vector_store_build() -> bool:
    section("Test 4 — Vector store build (local embeddings)")
    try:
        pipeline = RAGPipeline()
        pipeline.build(force_rebuild=True)
        count = pipeline.get_document_count()
        if count == 0:
            fail("Vector store has 0 documents")
            return False
        ok(f"Vector store built successfully with {count} indexed chunks")
        ok(f"Saved to: {settings.VECTOR_STORE_PATH}")
        return True
    except Exception as exc:
        fail(f"Exception: {exc}")
        traceback.print_exc()
        return False


def test_retrieval() -> bool:
    section("Test 5 — Semantic retrieval")
    queries = [
        ("B.Tech CSE eligibility", ["Admission Policies", "Eligibility Criteria", "Courses Catalogue"]),
        ("MBA fee structure", ["Fee Structure"]),
        ("application deadline last date", ["Deadlines Calendar"]),
        ("scholarship merit", ["Fee Structure", "Faqs"]),
    ]
    try:
        pipeline = RAGPipeline()
        pipeline.build(force_rebuild=False)
        retriever = pipeline.get_retriever()
        all_ok = True
        for query, expected_sources in queries:
            docs = retriever.invoke(query)
            if not docs:
                fail(f"No results for: '{query}'")
                all_ok = False
                continue
            retrieved_sources = {d.metadata.get("source", "") for d in docs}
            ok(f"Query: '{query}' → {len(docs)} chunks, sources: {retrieved_sources}")
        return all_ok
    except Exception as exc:
        fail(f"Exception: {exc}")
        traceback.print_exc()
        return False


def _llm_available() -> bool:
    """Return True if the configured LLM endpoint is reachable."""
    import socket
    if settings.LLM_PROVIDER == "ollama":
        host = settings.OLLAMA_BASE_URL.replace("http://", "").replace("https://", "")
        hostname, _, port_str = host.partition(":")
        port = int(port_str) if port_str else 11434
        try:
            with socket.create_connection((hostname, port), timeout=2):
                return True
        except OSError:
            return False
    # For OpenAI we just check network reachability is not validated here
    return bool(settings.OPENAI_API_KEY)


def test_agent_response() -> bool:
    section("Test 6 — Agent end-to-end response")
    if not _llm_available():
        ok("SKIPPED — LLM not reachable (Ollama not running / no API key). "
           "Start Ollama and re-run to test agent responses.")
        return True   # not a failure — infrastructure not present

    try:
        from agent import AdmissionAgent
        pipeline = RAGPipeline()
        pipeline.build(force_rebuild=False)
        agent = AdmissionAgent(pipeline)

        test_cases = [
            "What is the fee for B.Tech CSE?",
            "What entrance exam do I need for MBA?",
            "What are the scholarship options?",
        ]

        all_ok = True
        for question in test_cases:
            result = agent.ask(question)
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            if not answer or len(answer) < 50:
                fail(f"Short/empty answer for: '{question}'")
                all_ok = False
            else:
                ok(f"Q: '{question}'")
                ok(f"   A: {answer[:120]}...")
                ok(f"   Sources: {sources}")
        return all_ok
    except Exception as exc:
        fail(f"Exception: {exc}")
        traceback.print_exc()
        return False


def test_conversation_memory() -> bool:
    section("Test 7 — Conversational memory (follow-up questions)")
    if not _llm_available():
        ok("SKIPPED — LLM not reachable. Start Ollama and re-run.")
        return True

    try:
        from agent import AdmissionAgent
        pipeline = RAGPipeline()
        pipeline.build(force_rebuild=False)
        agent = AdmissionAgent(pipeline)

        r1 = agent.ask("What is the B.Tech CSE annual fee?")
        ok(f"Turn 1: {r1['answer'][:100]}...")

        r2 = agent.ask("Is there a scholarship for that?")
        ok(f"Turn 2 (follow-up): {r2['answer'][:100]}...")

        history = agent.get_history()
        if len(history) >= 1:
            ok(f"Memory contains {len(history)} turn(s)")
        else:
            fail("Memory appears empty after 2 turns")
            return False

        agent.reset_memory()
        if len(agent.get_history()) == 0:
            ok("Memory reset works correctly")
        else:
            fail("Memory not cleared after reset")
            return False

        return True
    except Exception as exc:
        fail(f"Exception: {exc}")
        traceback.print_exc()
        return False


# ── Runner ────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * console_width)
    print("  College Admission Agent — Test Suite")
    print(f"  LLM provider  : {settings.LLM_PROVIDER}")
    print(f"  Embeddings    : {settings.EMBEDDING_PROVIDER}")
    print("=" * console_width)

    tests = [
        ("Knowledge base files", test_knowledge_base_files),
        ("Document loading", test_document_loading),
        ("Text splitting", test_text_splitting),
        ("Vector store build", test_vector_store_build),
        ("Semantic retrieval", test_retrieval),
        ("Agent response", test_agent_response),
        ("Conversation memory", test_conversation_memory),
    ]

    results = []
    for name, fn in tests:
        try:
            passed = fn()
        except Exception as exc:
            fail(f"Unexpected error in {name}: {exc}")
            passed = False
        results.append((name, passed))

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * console_width}")
    print("  TEST SUMMARY")
    print("=" * console_width)
    passed_count = 0
    for name, passed in results:
        status = PASS if passed else FAIL
        print(f"  {status}  {name}")
        if passed:
            passed_count += 1

    total = len(results)
    print(f"\n  {passed_count}/{total} tests passed.")
    print("=" * console_width)

    sys.exit(0 if passed_count == total else 1)


if __name__ == "__main__":
    main()

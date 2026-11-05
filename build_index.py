"""
build_index.py — Standalone script to (re)build the FAISS vector index.

Run this once after adding or updating knowledge base documents:
    python build_index.py

Options:
    --force   Rebuild even if the index already exists.
    --stats   Print detailed chunk statistics after building.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich import box

from config import settings
from rag_pipeline import RAGPipeline, load_documents, split_documents

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the FAISS knowledge base index.")
    parser.add_argument("--force", action="store_true", help="Rebuild even if index exists.")
    parser.add_argument("--stats", action="store_true", help="Print chunk statistics.")
    args = parser.parse_args()

    console.print("\n[bold cyan]College Admission Agent — Index Builder[/bold cyan]")
    console.print(f"  Knowledge base : [bold]{settings.KNOWLEDGE_BASE_DIR}[/bold]")
    console.print(f"  Vector store   : [bold]{settings.VECTOR_STORE_PATH}[/bold]")
    console.print(f"  Embeddings     : [bold]{settings.EMBEDDING_PROVIDER}[/bold]\n")

    # ── Stats pass (before building) ─────────────────────────────────────────
    if args.stats:
        console.print("[bold]Loading documents for statistics…[/bold]")
        docs = load_documents(settings.KNOWLEDGE_BASE_DIR)
        chunks = split_documents(docs)

        table = Table(title="Document Statistics", box=box.SIMPLE)
        table.add_column("Source", style="bold")
        table.add_column("Chunks", justify="right")
        table.add_column("Total chars", justify="right")

        by_source: dict = {}
        for chunk in chunks:
            src = chunk.metadata.get("source", "Unknown")
            by_source.setdefault(src, {"count": 0, "chars": 0})
            by_source[src]["count"] += 1
            by_source[src]["chars"] += len(chunk.page_content)

        for src, info in sorted(by_source.items()):
            table.add_row(src, str(info["count"]), f"{info['chars']:,}")

        total_chunks = sum(v["count"] for v in by_source.values())
        total_chars = sum(v["chars"] for v in by_source.values())
        table.add_row(
            "[bold]TOTAL[/bold]",
            f"[bold]{total_chunks}[/bold]",
            f"[bold]{total_chars:,}[/bold]",
        )
        console.print(table)
        console.print()

    # ── Build ─────────────────────────────────────────────────────────────────
    index_exists = (settings.VECTOR_STORE_PATH / "index.faiss").exists()
    if index_exists and not args.force:
        console.print(
            "[bold yellow]Index already exists.[/bold yellow] "
            "Use [bold]--force[/bold] to rebuild.\n"
        )
        pipeline = RAGPipeline()
        with console.status("Loading existing index…", spinner="dots"):
            pipeline.build(force_rebuild=False)
        console.print(
            f"[bold green]✓[/bold green] Loaded — "
            f"[bold]{pipeline.get_document_count()}[/bold] chunks in index."
        )
        return

    start = time.time()
    pipeline = RAGPipeline()

    with console.status("[bold yellow]Building index…[/bold yellow]", spinner="dots"):
        pipeline.build(force_rebuild=True)

    elapsed = time.time() - start
    doc_count = pipeline.get_document_count()

    console.print(
        f"[bold green]✓[/bold green] Index built successfully in "
        f"[bold]{elapsed:.1f}s[/bold] — "
        f"[bold]{doc_count}[/bold] chunks indexed.\n"
    )


if __name__ == "__main__":
    main()

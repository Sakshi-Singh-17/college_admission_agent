"""
main.py — Interactive CLI for the College Admission Agent.

Run:
    python main.py            # interactive chat
    python main.py --rebuild  # force re-index knowledge base then chat
    python main.py --demo     # run a non-interactive demo with preset questions
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import List

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text
from rich import print as rprint

from config import settings
from rag_pipeline import RAGPipeline
from agent import AdmissionAgent

console = Console()

# ── UI helpers ────────────────────────────────────────────────────────────────

BANNER = """
╔═══════════════════════════════════════════════════════════════════╗
║          🎓  GREENFIELD UNIVERSITY — ADMISSION ASSISTANT          ║
║             Powered by RAG (Retrieval-Augmented Generation)       ║
╚═══════════════════════════════════════════════════════════════════╝
"""

COMMANDS = {
    "/help":    "Show this help message",
    "/reset":   "Clear conversation history and start fresh",
    "/history": "Show current conversation history",
    "/sources": "Toggle display of source documents",
    "/config":  "Show current configuration",
    "/quit":    "Exit the application",
}

DEMO_QUESTIONS: List[str] = [
    "What are the eligibility criteria for B.Tech CSE?",
    "What is the fee for MBA Finance, and are there scholarships available?",
    "What is the last date to apply for undergraduate programmes?",
    "I scored 62% in 12th with PCM. Which engineering branches can I apply to?",
    "How does the doctoral admission process work?",
    "What entrance exam do I need for MBA, and what is the minimum score required?",
    "Can a commerce student apply for BBA? What documents are needed?",
    "What is the refund policy if I withdraw after paying the fee?",
]


def print_banner() -> None:
    console.print(f"[bold cyan]{BANNER}[/bold cyan]")


def print_help() -> None:
    table = Table(title="Available Commands", box=box.SIMPLE, show_header=True)
    table.add_column("Command", style="bold yellow", no_wrap=True)
    table.add_column("Description", style="white")
    for cmd, desc in COMMANDS.items():
        table.add_row(cmd, desc)
    console.print(table)
    console.print()


def print_config() -> None:
    panel = Panel(
        settings.display(),
        title="[bold]Current Configuration[/bold]",
        border_style="dim",
        padding=(0, 1),
    )
    console.print(panel)


def print_answer(result: dict, show_sources: bool) -> None:
    """Pretty-print the agent's answer."""
    answer_md = Markdown(result["answer"])
    console.print()
    console.print(
        Panel(
            answer_md,
            title="[bold green]🎓 Admission Assistant[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )

    if show_sources and result["sources"]:
        sources_text = "  •  " + "\n  •  ".join(result["sources"])
        console.print(
            Panel(
                f"[dim]{sources_text}[/dim]",
                title="[dim]📄 Sources Consulted[/dim]",
                border_style="dim",
                padding=(0, 1),
            )
        )
    console.print()


def print_history(agent: AdmissionAgent) -> None:
    history = agent.get_history()
    if not history:
        console.print("[dim]No conversation history yet.[/dim]\n")
        return
    console.print(Rule("[bold]Conversation History[/bold]"))
    for i, (human, ai) in enumerate(history, 1):
        console.print(f"[bold cyan]Turn {i} — You:[/bold cyan]")
        console.print(f"  {human}")
        console.print(f"[bold green]Turn {i} — Assistant:[/bold green]")
        console.print(Markdown(ai))
        console.print()


# ── Initialisation ────────────────────────────────────────────────────────────

def initialise(force_rebuild: bool = False) -> AdmissionAgent:
    """Load (or build) the vector store and initialise the agent."""

    console.print()
    console.print(Rule("[bold]Initialising Admission Agent[/bold]"))

    # Validate config
    try:
        settings.validate()
    except EnvironmentError as exc:
        console.print(f"[bold red]Configuration Error:[/bold red] {exc}")
        sys.exit(1)

    print_config()

    # Build / load vector store
    pipeline = RAGPipeline()
    index_exists = (settings.VECTOR_STORE_PATH / "index.faiss").exists()
    action = "Rebuilding" if (force_rebuild or not index_exists) else "Loading"

    with console.status(
        f"[bold yellow]{action} knowledge base index…[/bold yellow]",
        spinner="dots",
    ):
        pipeline.build(force_rebuild=force_rebuild)

    doc_count = pipeline.get_document_count()
    console.print(
        f"[bold green]✓[/bold green] Knowledge base ready — "
        f"[bold]{doc_count}[/bold] chunks indexed."
    )

    # Create agent
    with console.status("[bold yellow]Loading language model…[/bold yellow]", spinner="dots"):
        agent = AdmissionAgent(pipeline)

    console.print("[bold green]✓[/bold green] Admission Agent is ready!\n")
    return agent


# ── Interactive chat loop ─────────────────────────────────────────────────────

def chat_loop(agent: AdmissionAgent) -> None:
    show_sources = True

    console.print(
        Panel(
            "[bold]Welcome![/bold] Ask me anything about admissions at Greenfield University.\n"
            "Type [bold yellow]/help[/bold yellow] to see available commands, "
            "or [bold yellow]/quit[/bold yellow] to exit.",
            border_style="cyan",
            padding=(0, 1),
        )
    )
    console.print()

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Session ended.[/dim]")
            break

        if not user_input:
            continue

        # ── Handle commands ──────────────────────────────────────────────────
        cmd = user_input.lower()

        if cmd in ("/quit", "/exit", "/q"):
            console.print("[dim]Thank you for using the GFU Admission Assistant. Good luck![/dim]")
            break

        if cmd == "/help":
            print_help()
            continue

        if cmd == "/reset":
            agent.reset_memory()
            console.print("[bold yellow]Conversation history cleared.[/bold yellow]\n")
            continue

        if cmd == "/history":
            print_history(agent)
            continue

        if cmd == "/sources":
            show_sources = not show_sources
            state = "ON" if show_sources else "OFF"
            console.print(f"[bold yellow]Source display turned {state}.[/bold yellow]\n")
            continue

        if cmd == "/config":
            print_config()
            continue

        # ── Ask the agent ────────────────────────────────────────────────────
        with console.status("[bold yellow]Thinking…[/bold yellow]", spinner="dots"):
            try:
                result = agent.ask(user_input)
            except Exception as exc:
                console.print(f"\n[bold red]Error:[/bold red] {exc}\n")
                continue

        print_answer(result, show_sources)


# ── Demo mode ─────────────────────────────────────────────────────────────────

def demo_mode(agent: AdmissionAgent) -> None:
    """Run preset questions non-interactively for testing."""
    console.print(Rule("[bold cyan]Demo Mode — Preset Questions[/bold cyan]"))
    console.print(
        "[dim]Running a set of sample admission queries against the knowledge base…[/dim]\n"
    )

    for i, question in enumerate(DEMO_QUESTIONS, 1):
        console.print(f"[bold cyan]Q{i}:[/bold cyan] {question}")

        with console.status("[bold yellow]Thinking…[/bold yellow]", spinner="dots"):
            try:
                result = agent.ask(question)
            except Exception as exc:
                console.print(f"  [bold red]Error:[/bold red] {exc}\n")
                continue

        print_answer(result, show_sources=True)
        time.sleep(0.5)   # brief pause between questions

    console.print(Rule("[bold green]Demo Complete[/bold green]"))


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="College Admission Agent — RAG-powered assistant for Greenfield University"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force re-index the knowledge base before starting.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run non-interactive demo with preset questions.",
    )
    args = parser.parse_args()

    print_banner()
    agent = initialise(force_rebuild=args.rebuild)

    if args.demo:
        demo_mode(agent)
    else:
        chat_loop(agent)


if __name__ == "__main__":
    main()

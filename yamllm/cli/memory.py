"""
Memory management CLI commands for YAMLLM.

This module contains memory and vector store related CLI commands.
"""

import argparse
import os
from rich.console import Console
from rich.prompt import Confirm

console = Console()


def migrate_index(args: argparse.Namespace) -> int:
    """Migrate or purge FAISS vector index."""
    try:
        import faiss
    except Exception as e:
        console.print("[red]FAISS is not installed. Install 'faiss-cpu' to manage vector indexes.[/red]")
        return 1
    
    store_path = args.store_path
    index_path = os.path.join(store_path, "faiss_index.idx")
    metadata_path = os.path.join(store_path, "metadata.pkl")

    if not os.path.exists(index_path):
        console.print(f"[yellow]No index found at {index_path}. Nothing to migrate.[/yellow]")
        return 0

    try:
        index = faiss.read_index(index_path)
        dim = getattr(index, "d", None)
        console.print(f"[cyan]Found index: {index_path} (dimension={dim})[/cyan]")
    except Exception as e:
        console.print(f"[red]Failed to read index: {e}[/red]")
        return 1

    mismatched = args.expect_dim is not None and dim is not None and dim != args.expect_dim
    if mismatched:
        console.print(
            f"[yellow]Dimension mismatch: index={dim}, expected={args.expect_dim}. "
            f"This indicates the embedding model has changed.[/yellow]"
        )

    if args.purge or (mismatched and Confirm.ask("Purge incompatible index/metadata now?", default=False)):
        try:
            if os.path.exists(index_path):
                os.remove(index_path)
                console.print(f"[green]✓ Deleted {index_path}[/green]")
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
                console.print(f"[green]✓ Deleted {metadata_path}[/green]")
            console.print("[green]Index purged. It will be rebuilt on next use.[/green]")
        except Exception as e:
            console.print(f"[red]✗ Failed to purge index: {e}[/red]")
            return 1
    else:
        console.print("[yellow]No changes made. Use --purge to delete the index.[/yellow]")

    return 0


def setup_memory_commands(subparsers):
    """Set up memory-related CLI commands."""
    # Migrate index command
    mig = subparsers.add_parser("migrate-index", help="Inspect or purge FAISS index for a vector store")
    mig.add_argument("--store-path", required=True, help="Path to vector store directory")
    mig.add_argument("--expect-dim", type=int, default=None, help="Expected embedding dimension")
    mig.add_argument("--purge", action="store_true", help="Delete index and metadata to rebuild from scratch")
    mig.set_defaults(func=migrate_index)

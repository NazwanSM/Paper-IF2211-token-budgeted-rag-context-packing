"""Development import shim for the src-layout package.

The implementation lives in ``src/rag_context_packing``. This small root package
keeps ``python -m rag_context_packing.cli`` usable from the repository root
without requiring an editable install first.
"""

from __future__ import annotations

from pathlib import Path

SRC_PACKAGE = Path(__file__).resolve().parent.parent / "src" / "rag_context_packing"
if SRC_PACKAGE.is_dir():
    __path__.append(str(SRC_PACKAGE))

from rag_context_packing.models import Chunk, SelectionResult

__all__ = ["Chunk", "SelectionResult"]

"""Institutional negotiation memory.

Qdrant-backed storage and semantic recall of :class:`~precedent.models.Precedent`
records. This layer owns embeddings, collection lifecycle, upserts, and
similarity search — the mechanism by which the institution *remembers* how it
negotiated similar clauses before.
"""

from __future__ import annotations

__all__: list[str] = []

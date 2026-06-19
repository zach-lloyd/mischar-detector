"""
Stage 4: Retrieval via chunking, embedding, and semantic similarity.

This is the pipeline's RAG (retrieval-augmented generation) stage. Given
a case opinion and a claim, it:

1. Chunks the opinion into paragraph-grouped segments.
2. Embeds each chunk and the claim via Voyage's voyage-law-2 model.
3. Ranks chunks by cosine similarity to the claim.
4. Returns the top-K most relevant chunks for the classifier.

This stage is what makes the classifier tractable — instead of feeding
a 50-page opinion to the LLM, we feed it the 5 most relevant chunks.
"""

from __future__ import annotations

import re

import numpy as np

from mischar.cache import Cache
from mischar.logging import get_logger
from mischar.models.embedding import EmbeddingClient
from mischar.types import (
    Chunk,
    ChunkEmbedding,
    Embedding,
    RetrievalResult,
)

log = get_logger("retrieve")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_opinion(
    text: str,
    max_tokens: int = 1200,
    overlap_paragraphs: int = 1,
) -> list[Chunk]:
    """
    Split opinion text into paragraph-grouped chunks.

    Paragraphs are packed consecutively into chunks until adding the
    next paragraph would exceed ``max_tokens``. The last paragraph(s) of
    each chunk overlap with the first paragraph(s) of the next chunk,
    ensuring context continuity at chunk boundaries.

    Token counting uses a rough heuristic (chars / 4) rather than a
    real tokenizer — good enough for chunking purposes and avoids a
    tokenizer dependency.

    Args:
        text: The full opinion text.
        max_tokens: Maximum approximate tokens per chunk.
        overlap_paragraphs: Number of paragraphs to overlap between
            consecutive chunks. Provides context continuity.

    Returns:
        A list of ``Chunk`` objects with text, token count, index, and
        paragraph range.
    """
    # Split into paragraphs on double newlines. Filter out empty strings
    # that result from consecutive newlines.
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    if not paragraphs:
        return []

    # Pre-split any single paragraph that already exceeds the token cap, so
    # the packing loop below can never emit an oversized chunk. Opinions
    # whose text has few "\n\n" breaks (e.g. HTML-stripped text) would
    # otherwise collapse into one giant "paragraph" and become a single
    # massive chunk.
    expanded_paragraphs: list[str] = []
    for para in paragraphs:
        expanded_paragraphs.extend(_split_long_paragraph(para, max_tokens))
    paragraphs = expanded_paragraphs

    chunks = []
    chunk_index = 0
    para_start = 0  # Index of the first paragraph in the current chunk.

    while para_start < len(paragraphs):
        # Pack paragraphs into this chunk until we'd exceed the token limit.
        current_paras = []
        current_tokens = 0
        para_end = para_start

        while para_end < len(paragraphs):
            para_text = paragraphs[para_end]
            para_tokens = _estimate_tokens(para_text)

            # If adding this paragraph would exceed the limit and we
            # already have at least one paragraph, stop here. If we do not 
            # already have at least one paragraph, add just this paragraph as its
            # own oversized chunk. Given that voyage-law-2 has a 16k token context
            # window, it is highly unlikely that any single paragraph will be too
            # long to be its own chunk.
            if current_tokens + para_tokens > max_tokens and current_paras:
                break

            current_paras.append(para_text)
            current_tokens += para_tokens
            para_end += 1

        # Build the chunk from the collected paragraphs.
        chunk_text = "\n\n".join(current_paras)
        chunks.append(Chunk(
            text=chunk_text,
            token_count=current_tokens,
            chunk_index=chunk_index,
            paragraph_range=(para_start, para_end - 1),  # inclusive range
        ))

        chunk_index += 1

        # Advance to the next chunk, overlapping by the specified number
        # of paragraphs. This means the last N paragraphs of this chunk
        # become the first N paragraphs of the next chunk.
        para_start = para_end - overlap_paragraphs

        # Safety: if overlap would cause us to not advance at all
        # (single very long paragraph), force advancement.
        if para_start <= chunks[-1].paragraph_range[0]:
            para_start = para_end

    log.debug(
        "chunking_complete",
        total_paragraphs=len(paragraphs),
        chunks=len(chunks),
    )

    return chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def embed_chunks(
    chunks: list[Chunk],
    client: EmbeddingClient,
    cache: Cache,
) -> list[ChunkEmbedding]:
    """
    Embed each chunk via Voyage, with per-chunk caching.

    Checks the cache for each chunk individually (keyed by content hash)
    and only sends uncached chunks to the embedding API. This avoids
    re-embedding chunks when the same opinion is processed again.

    Args:
        chunks: The chunks to embed.
        client: Voyage embedding client.
        cache: Pipeline cache instance.

    Returns:
        A list of ``ChunkEmbedding`` objects in the same order as input.
    """
    results: list[ChunkEmbedding | None] = [None] * len(chunks)
    uncached_indices: list[int] = []
    uncached_texts: list[str] = []

    # Check cache for each chunk.
    for i, chunk in enumerate(chunks):
        cache_key = Cache.make_key("chunk_embedding", chunk.text)
        cached_embedding = cache.get("embedding", cache_key)

        if cached_embedding is not None:
            results[i] = ChunkEmbedding(chunk=chunk, embedding=cached_embedding)
        else:
            uncached_indices.append(i)
            uncached_texts.append(chunk.text)

    # Embed uncached chunks in a single batch call.
    if uncached_texts:
        log.info("embed_chunks_api_call", count=len(uncached_texts))
        embeddings = client.embed(uncached_texts, input_type="document")

        # strict=True ensures the API returned exactly as many embeddings as we
        # sent texts, and that if there's a mismatch, it gets raised immediately.
        for idx, embedding in zip(uncached_indices, embeddings, strict=True):
            chunk = chunks[idx]
            results[idx] = ChunkEmbedding(chunk=chunk, embedding=embedding)

            # Cache each embedding individually.
            cache_key = Cache.make_key("chunk_embedding", chunk.text)
            cache.set("embedding", cache_key, embedding)

    return results  # type: ignore[return-value]


def embed_claim(claim: str, client: EmbeddingClient, cache: Cache) -> Embedding:
    """
    Embed a claim string, with caching.

    Claims are embedded as "query" type for asymmetric retrieval
    (Voyage optimizes query vs document embeddings differently).

    Args:
        claim: The claim text to embed.
        client: Voyage embedding client.
        cache: Pipeline cache instance.

    Returns:
        The embedding vector (list of 1024 floats).
    """
    cache_key = Cache.make_key("claim_embedding", claim)
    cached = cache.get("embedding", cache_key)

    if cached is not None:
        return cached

    log.debug("embed_claim_api_call")
    embeddings = client.embed([claim], input_type="query")
    embedding = embeddings[0]

    cache.set("embedding", cache_key, embedding)

    return embedding


# ---------------------------------------------------------------------------
# Similarity + retrieval
# ---------------------------------------------------------------------------


def retrieve_top_k(
    claim_embedding: Embedding,
    chunk_embeddings: list[ChunkEmbedding],
    k: int = 5
) -> RetrievalResult:
    """
    Select the top-K most relevant chunks for a claim.

    Computes cosine similarity between the claim and each chunk and returns the 
    top-K chunks.

    Fallback: if the opinion has fewer than 3 chunks, we return all
    chunks rather than selecting top-K — the opinion is short enough
    to use as full context.

    Args:
        claim_embedding: The embedded claim vector.
        chunk_embeddings: Embedded chunks from the opinion.
        k: Number of top chunks to return.

    Returns:
        A ``RetrievalResult`` with the selected chunks and their scores.
    """
    n_chunks = len(chunk_embeddings)

    # Short opinion fallback: if fewer than 3 chunks, return everything.
    # The opinion is short enough to be used as full context.
    if n_chunks <= 3:
        log.debug("retrieve_short_opinion", n_chunks=n_chunks)
        
        return RetrievalResult(
            chunks=[ce.chunk for ce in chunk_embeddings],
            scores=[1.0] * n_chunks,
        )

    # Compute cosine similarity between the claim and each chunk.
    claim_vec = np.array(claim_embedding)
    scores = []
    for ce in chunk_embeddings:
        chunk_vec = np.array(ce.embedding)
        similarity = _cosine_similarity(claim_vec, chunk_vec)
        scores.append(similarity)

    # Select the top-K chunks by (boosted) score.
    # argsort returns ascending order; we want descending, so negate or reverse.
    score_array = np.array(scores)
    top_indices = np.argsort(score_array)[::-1][:k]

    # Sort selected indices by their position in the opinion so the
    # classifier sees chunks in reading order, not relevance order.
    top_indices = sorted(top_indices)

    selected_chunks = [chunk_embeddings[i].chunk for i in top_indices]
    selected_scores = [scores[i] for i in top_indices]

    log.info(
        "retrieve_complete",
        k=k,
        n_chunks=n_chunks,
        top_scores=[round(s, 3) for s in selected_scores],
    )

    return RetrievalResult(
        chunks=selected_chunks,
        scores=selected_scores,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: The first vector.
        b: The second vector.

    Returns: 
        A value between -1 and 1, where 1 means identical direction.
    """
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # Guard against zero-norm vectors (shouldn't happen with real
    # embeddings, but defensive).
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot / (norm_a * norm_b))


_SENTENCE_RE = re.compile(r"[^.!?]*[.!?]+(?:\s+|$)|[^.!?]+$")


def _pack_units(units: list[str], max_tokens: int) -> list[str]:
    """
    Greedily pack text units into pieces of at most ``max_tokens``.

    Args:
        units: Text units (sentences or words) to pack, in order.
        max_tokens: Maximum approximate tokens per packed piece.

    Returns:
        A list of packed pieces, each at or under the token limit
        (a single unit that exceeds the limit is emitted on its own).
    """
    pieces: list[str] = []
    current = ""

    for unit in units:
        unit = unit.strip()
        if not unit:
            continue

        candidate = f"{current} {unit}".strip() if current else unit
        if current and _estimate_tokens(candidate) > max_tokens:
            pieces.append(current)
            current = unit
        else:
            current = candidate

    if current:
        pieces.append(current)

    return pieces


def _split_long_paragraph(paragraph: str, max_tokens: int) -> list[str]:
    """
    Split a paragraph that exceeds ``max_tokens`` into smaller pieces.

    Splits on sentence boundaries and greedily repacks them; any single
    sentence still over the limit is further split on word boundaries.
    Paragraphs already within the limit are returned unchanged. This
    prevents a single oversized "paragraph" from becoming one giant chunk.

    Args:
        paragraph: The paragraph text to (possibly) split.
        max_tokens: Maximum approximate tokens per resulting piece.

    Returns:
        A list of pieces, each at or under the token limit.
    """
    if _estimate_tokens(paragraph) <= max_tokens:
        return [paragraph]

    sentences = [s for s in _SENTENCE_RE.findall(paragraph) if s.strip()]

    # Break any sentence that is itself over the limit into words first.
    units: list[str] = []
    for sentence in sentences:
        if _estimate_tokens(sentence) <= max_tokens:
            units.append(sentence)
        else:
            units.extend(_pack_units(sentence.split(), max_tokens))

    return _pack_units(units, max_tokens)


def _estimate_tokens(text: str) -> int:
    """
    Rough token count estimate.

    Uses the heuristic of ~4 characters per token, which is a reasonable
    average for English text. We don't import a real tokenizer here to
    avoid adding a heavy dependency just for chunking.

    Args:
        text: The text to be estimated.
    
    Returns:
        The estimated number of tokens in the text assuming 4 characters per
        token.
    """
    return max(1, len(text) // 4)

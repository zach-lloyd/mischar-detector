"""
Embedding client for Voyage AI's voyage-law-2 model.

Provides text embeddings optimized for legal content. Used by the retrieve
stage to embed both opinion chunks (as documents) and claims (as queries)
for semantic similarity matching.
"""

from __future__ import annotations

from typing import Literal

from mischar.logging import get_logger
from mischar.models.client import ModelClientError, retry_with_backoff
from mischar.types import Embedding

log = get_logger("embedding")

# Voyage enforces two per-batch limits: a maximum total token count
# (120,000 for voyage-law-2) and a maximum number of texts (1,000). We
# stay conservatively under both — the token figure is a chars/4 estimate,
# so the headroom absorbs estimation error.
_MAX_TOKENS_PER_BATCH = 100_000
_MAX_TEXTS_PER_BATCH = 1_000


def _estimate_tokens(text: str) -> int:
    """
    Rough token estimate (~4 characters per token).

    Matches the heuristic used by the chunker so batch sizing is
    consistent across the pipeline.

    Args:
        text: The text to estimate.

    Returns:
        Estimated token count (at least 1).
    """
    return max(1, len(text) // 4)


class EmbeddingClient:
    """
    Client for Voyage AI's embedding API.

    Voyage's ``voyage-law-2`` model produces 1024-dimensional embeddings
    specifically trained on legal text, making it well-suited for matching
    legal claims against opinion chunks.

    Args:
        api_key: Voyage AI API key.
        model: Embedding model identifier. Defaults to ``voyage-law-2``.
    """

    def __init__(self, api_key: str, model: str = "voyage-law-2") -> None:
        try:
            import voyageai
        except ImportError as exc:
            raise ImportError(
                "EmbeddingClient requires the 'voyageai' package. "
                "Install with: pip install -e '.[local]'"
            ) from exc

        self._client = voyageai.Client(api_key=api_key)
        self._model = model

        log.info("embedding_client_initialized", model=model)


    def embed(
        self,
        texts: list[str],
        input_type: Literal["document", "query"],
    ) -> list[Embedding]:
        """
        Embed a list of texts.

        Voyage distinguishes between document and query embeddings for
        better retrieval performance. Opinion chunks should be embedded
        as "document"; claims should be embedded as "query".

        Args:
            texts: The texts to embed.
            input_type: Either "document" (for opinion chunks being
                indexed) or "query" (for claims being searched).

        Returns:
            A list of embedding vectors (each a list of 1024 floats),
            one per input text, in the same order.

        Raises:
            ModelClientError: If the Voyage API is unreachable after
                retries.
        """
        if not texts:
            return []

        # Split into sub-batches that respect Voyage's per-batch token and
        # count limits, embed each, and concatenate in order. A single
        # opinion can produce 100+ chunks whose combined tokens exceed the
        # 120k batch cap, so one big request would otherwise fail.
        embeddings: list[Embedding] = []
        batch: list[str] = []
        batch_tokens = 0

        for text in texts:
            text_tokens = _estimate_tokens(text)

            # Flush the current batch before it would exceed either limit.
            if batch and (
                batch_tokens + text_tokens > _MAX_TOKENS_PER_BATCH
                or len(batch) >= _MAX_TEXTS_PER_BATCH
            ):
                embeddings.extend(self._embed_batch(batch, input_type))
                batch = []
                batch_tokens = 0

            batch.append(text)
            batch_tokens += text_tokens

        if batch:
            embeddings.extend(self._embed_batch(batch, input_type))

        return embeddings


    def _embed_batch(
        self,
        texts: list[str],
        input_type: Literal["document", "query"],
    ) -> list[Embedding]:
        """
        Embed a single batch that already fits within Voyage's limits.

        Args:
            texts: The texts for one API call (within token/count caps).
            input_type: Either "document" or "query".

        Returns:
            The embedding vectors for this batch, in input order.

        Raises:
            ModelClientError: If the Voyage API fails after retries.
        """
        def _call():
            return self._client.embed(
                texts=texts,
                model=self._model,
                input_type=input_type,
            )

        try:
            result = retry_with_backoff(
                _call,
                retryable_exceptions=(Exception,),
                context=f"voyage embed ({self._model}, {len(texts)} texts)",
            )
        except Exception as exc:
            raise ModelClientError(
                backend="voyage",
                message=f"Embedding call failed: {exc}",
                cause=exc,
            ) from exc

        return result.embeddings

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

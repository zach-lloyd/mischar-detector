"""
Pipeline orchestrator — core five-stage classification pipeline.

Wires parse → resolve → attribute → retrieve → classify into a single
``process_passage`` method that produces one ``CitationResult`` per
citation found in the input text. Each stage can also be called
independently for testing and retrieval verification.

All external dependencies (model clients, embedding client, CourtListener
client, cache) are injected at construction time so the Pipeline itself
has no hidden coupling to infrastructure.
"""

from __future__ import annotations

from mischar.cache import Cache
from mischar.config import Config
from mischar.logging import get_logger
from mischar.models.client import ModelClient
from mischar.models.embedding import EmbeddingClient
from mischar.stages.attribute import attribute_claim
from mischar.stages.classify import classify as classify_citation
from mischar.stages.parse import parse_citations
from mischar.stages.resolve import CourtListenerClient, resolve_citation
from mischar.stages.retrieve import (
    chunk_opinion,
    embed_chunks,
    embed_claim,
    retrieve_top_k,
)
from mischar.types import (
    Abstention,
    AttributedClaim,
    CitationResult,
    Classification,
    ParsedCitation,
    ResolvedCase,
    RetrievalResult,
)

log = get_logger("pipeline")


class Pipeline:
    """
    Five-stage legal citation mischaracterization detector.

    Processes a passage of text from a legal brief, extracts each citation,
    resolves it against CourtListener, attributes the claimed holding via
    LLM, retrieves relevant chunks from the opinion, and classifies whether
    the citation is accurately characterized.

    Each stage can short-circuit into an ``Abstention`` when a meaningful
    classification isn't possible (case not found, no clear claim, etc.).
    Infrastructure errors (network failures, API errors) propagate as
    exceptions — they are not abstentions.

    Args:
        config: Application configuration (retrieval params, prompt versions,
            model selection, etc.).
        attribution_client: LLM client for the attribution stage (Stage 3).
        classifier_client: LLM client for the classification stage (Stage 5).
        embedding_client: Voyage embedding client for the retrieval stage.
        courtlistener_client: CourtListener API client for the resolve stage.
        cache: Pipeline cache instance.
    """

    def __init__(
        self,
        config: Config,
        attribution_client: ModelClient,
        classifier_client: ModelClient,
        embedding_client: EmbeddingClient,
        courtlistener_client: CourtListenerClient,
        cache: Cache,
    ) -> None:
        self._config = config
        self._attribution_client = attribution_client
        self._classifier_client = classifier_client
        self._embedding_client = embedding_client
        self._courtlistener_client = courtlistener_client
        self._cache = cache

        log.info(
            "pipeline_initialized",
            attribution_model=attribution_client.name,
            classifier_model=classifier_client.name,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_passage(self, passage: str) -> list[CitationResult]:
        """
        Run the full five-stage pipeline on a passage.

        Extracts every citation from the passage, then processes each one
        independently through resolve → attribute → retrieve → classify.
        Each citation produces exactly one ``CitationResult`` — either a
        classification or an abstention.

        If no citations are found in the passage, returns an empty list.
        The caller (CLI or eval harness) is responsible for reporting the
        "no citations found" condition to the user.

        Args:
            passage: Text from a legal brief containing one or more
                citations. Typically a paragraph or multi-paragraph excerpt.

        Returns:
            A list of ``CitationResult`` objects, one per citation found,
            in the order they appear in the passage. Empty list if no
            citations could be parsed.
        """
        # Stage 1: Parse.
        citations = self.parse(passage)

        if not citations:
            log.info("pipeline_no_citations", passage_preview=passage[:100])

            return []

        log.info("pipeline_start", n_citations=len(citations))

        # Process each citation through the remaining four stages.
        results = []
        for citation in citations:
            result = self._process_single_citation(passage, citation)
            results.append(result)

        log.info(
            "pipeline_complete",
            n_results=len(results),
            n_abstentions=sum(1 for r in results if r.abstained),
        )

        return results

    # ------------------------------------------------------------------
    # Individual stage methods (separately callable for testing)
    # ------------------------------------------------------------------

    def parse(self, passage: str) -> list[ParsedCitation]:
        """
        Stage 1: Extract legal citations from a passage.

        Delegates to eyecite via ``stages.parse.parse_citations``. Only
        full case citations are returned — short-form ("Id.") and supra
        references are skipped.

        Args:
            passage: The text to search for citations.

        Returns:
            A list of ``ParsedCitation`` objects. Empty if no full case
            citations are found.
        """
        return parse_citations(passage)


    def resolve(self, citation: ParsedCitation) -> ResolvedCase | Abstention:
        """
        Stage 2: Resolve a citation to its full opinion text.

        Queries CourtListener to find the case and fetch its opinion.
        Results are cached by citation components (volume, reporter, page).

        Args:
            citation: The parsed citation to resolve.

        Returns:
            A ``ResolvedCase`` with full opinion text, or an ``Abstention``
            with reason ``case-not-found`` or ``text-not-retrieved``.
        """
        return resolve_citation(
            citation=citation,
            client=self._courtlistener_client,
            cache=self._cache,
        )


    def attribute(
        self, passage: str, citation: ParsedCitation
    ) -> AttributedClaim | Abstention:
        """
        Stage 3: Extract the claim a passage attributes to a cited case.

        Calls the attribution LLM to identify the specific legal
        proposition the passage says the case stands for.

        Args:
            passage: The full passage text from the legal brief.
            citation: The parsed citation to attribute.

        Returns:
            An ``AttributedClaim`` with the extracted proposition, or an
            ``Abstention`` with reason ``attribution-failed``.
        """
        return attribute_claim(
            passage=passage,
            citation=citation,
            client=self._attribution_client,
            cache=self._cache,
            prompt_version=self._config.attribution_prompt_version,
        )


    def retrieve(
        self,
        case: ResolvedCase,
        claim: AttributedClaim,
    ) -> RetrievalResult | Abstention:
        """
        Stage 4: Retrieve relevant chunks from the opinion.

        Chunks the opinion text, embeds chunks and claim via Voyage,
        then selects the top-K most relevant chunks by cosine similarity.

        Args:
            case: The resolved case with full opinion text.
            claim: The attributed claim to match against.

        Returns:
            A ``RetrievalResult`` with the selected chunks and scores,
            or an ``Abstention`` with reason ``case-too-long`` if the
            retrieved context would be unmanageable (defensive — unlikely
            with current parameters).
        """
        # Chunk the opinion text.
        chunks = chunk_opinion(
            text=case.full_text,
            max_tokens=self._config.chunk_max_tokens,
            overlap_paragraphs=self._config.chunk_overlap_paragraphs,
        )

        if not chunks:
            return Abstention(
                reason="text-not-retrieved",
                details=f"Opinion text for {case.case_name} produced no chunks",
            )

        # Embed chunks and claim.
        chunk_embeddings = embed_chunks(
            chunks=chunks,
            client=self._embedding_client,
            cache=self._cache,
        )

        claim_embedding = embed_claim(
            claim=claim.claim_text,
            client=self._embedding_client,
            cache=self._cache,
        )

        # Select top-K chunks by cosine similarity.
        retrieval = retrieve_top_k(
            claim_embedding=claim_embedding,
            chunk_embeddings=chunk_embeddings,
            k=self._config.top_k,
        )

        return retrieval


    def classify(
        self,
        claim: AttributedClaim,
        retrieval: RetrievalResult,
        case: ResolvedCase,
    ) -> Classification:
        """
        Stage 5: Classify the relationship between claim and case text.

        Calls the classifier LLM with the claim and retrieved chunks to
        determine whether the claim accurately characterizes the case
        or mischaracterizes it.

        Args:
            claim: The attributed claim (what the brief says the case held).
            retrieval: The top-K retrieved chunks from the case opinion.
            case: The resolved case (used for case name in the prompt).

        Returns:
            A ``Classification`` with label, confidence, and supporting text.
        """
        return classify_citation(
            claim=claim,
            retrieval=retrieval,
            case=case,
            client=self._classifier_client,
            cache=self._cache,
            prompt_version=self._config.classification_prompt_version,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _process_single_citation(
        self,
        passage: str,
        citation: ParsedCitation,
    ) -> CitationResult:
        """
        Run stages 2-5 for a single citation and assemble the result.

        Each stage can produce an ``Abstention``, which short-circuits
        the remaining stages. Infrastructure errors (model failures,
        network errors) propagate as exceptions — the caller decides
        how to handle them.

        Args:
            passage: The full passage text (needed for attribution).
            citation: The parsed citation to process.

        Returns:
            A ``CitationResult`` with either a classification or an
            abstention.
        """
        log.info("pipeline_citation_start", cite=citation.raw_text)

        # Stage 2: Resolve.
        resolved = self.resolve(citation)
        if isinstance(resolved, Abstention):
            log.info(
                "pipeline_citation_abstained",
                cite=citation.raw_text,
                stage="resolve",
                reason=resolved.reason,
            )

            return CitationResult(
                citation=citation,
                resolved_case=None,
                claim=None,
                retrieval=None,
                classification=None,
                abstention=resolved,
                model_used=self._classifier_client.name,
            )

        # Stage 3: Attribute.
        claim = self.attribute(passage, citation)
        if isinstance(claim, Abstention):
            log.info(
                "pipeline_citation_abstained",
                cite=citation.raw_text,
                stage="attribute",
                reason=claim.reason,
            )

            return CitationResult(
                citation=citation,
                resolved_case=resolved,
                claim=None,
                retrieval=None,
                classification=None,
                abstention=claim,
                model_used=self._attribution_client.name,
            )

        # Stage 4: Retrieve.
        retrieval = self.retrieve(resolved, claim)
        if isinstance(retrieval, Abstention):
            log.info(
                "pipeline_citation_abstained",
                cite=citation.raw_text,
                stage="retrieve",
                reason=retrieval.reason,
            )

            return CitationResult(
                citation=citation,
                resolved_case=resolved,
                claim=claim,
                retrieval=None,
                classification=None,
                abstention=retrieval,
                model_used=self._classifier_client.name,
            )

        # Stage 5: Classify.
        classification = self.classify(claim, retrieval, resolved)

        log.info(
            "pipeline_citation_classified",
            cite=citation.raw_text,
            label=classification.label,
            confidence=round(classification.confidence, 3),
        )

        return CitationResult(
            citation=citation,
            resolved_case=resolved,
            claim=claim,
            retrieval=retrieval,
            classification=classification,
            abstention=None,
            model_used=self._classifier_client.name,
        )

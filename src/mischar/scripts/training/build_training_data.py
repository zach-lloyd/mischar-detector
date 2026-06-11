"""
Build fine-tuning data from the CaseHOLD accurate/mischaracterized pairs.

Takes the output of ``build_casehold_set.py`` and converts each example
into a ``prompt``/``completion`` record ready for the Modal training
scripts (``train_primary.py`` / ``train_secondary.py``):

1. Assigns train/val splits (pair members always share a split because
   they share a citation).
2. Resolves each entry's cited case via CourtListener (one resolution
   per entry — both pair members cite the same case).
3. Chunks and embeds the opinion, embeds each claim, and retrieves the
   top-K most relevant chunks — the same retrieval the deployed
   pipeline performs, so training inputs match inference inputs.
4. Builds the classification prompt and a label-only JSON completion.
5. Writes ``train.jsonl`` and ``val.jsonl``.

Entries are dropped as a pair when the case can't be resolved or text
isn't available, keeping the accurate/mischaracterized classes balanced.

Label-noise guard: for ACCURATE examples, if the best retrieval score
falls below ``--min-retrieval-score``, the pair is dropped — a low score
suggests the retrieved excerpts don't contain the holding, which would
teach the model that unsupported claims are "accurate". The default
threshold is 0.0 (disabled); the script prints a score distribution
summary so you can choose a sensible threshold and re-run (resolutions
and embeddings are cached, so re-runs are cheap).

Usage:
    python -m mischar.scripts.training.build_training_data \\
        --source data/processed/casehold.jsonl \\
        --output-dir data/training \\
        [--config src/mischar/config.yaml] \\
        [--val-ratio 0.15] \\
        [--min-retrieval-score 0.0] \\
        [--max-entries 100] \\
        [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv

from mischar.cache import Cache
from mischar.config import load_config, load_secrets
from mischar.data.datasets import load_casehold, validate_labels
from mischar.data.splits import assign_splits
from mischar.logging import configure_logging, get_logger
from mischar.models.embedding import EmbeddingClient
from mischar.prompts.classification import build_classification_prompt
from mischar.stages.parse import parse_citations
from mischar.stages.resolve import CourtListenerClient, resolve_citation
from mischar.stages.retrieve import (
    chunk_opinion,
    embed_chunks,
    embed_claim,
    retrieve_top_k,
)
from mischar.types import Abstention, EvalExample, ResolvedCase

log = get_logger("scripts.build_training_data")


# ---------------------------------------------------------------------------
# Entry grouping
# ---------------------------------------------------------------------------


def _group_into_pairs(
    examples: list[EvalExample],
) -> dict[str, list[EvalExample]]:
    """
    Group examples by their source CaseHOLD entry.

    Each entry should contribute exactly two examples (one accurate,
    one mischaracterized). Groups that aren't complete pairs are
    logged and excluded so the dataset stays balanced.

    Args:
        examples: Examples loaded from the CaseHOLD JSONL.

    Returns:
        A dict mapping source_entry_id to its [accurate, mischaracterized]
        examples.
    """
    groups: dict[str, list[EvalExample]] = defaultdict(list)

    for example in examples:
        entry_id = example.metadata.get("source_entry_id") or example.example_id
        groups[entry_id].append(example)

    complete = {}
    incomplete = 0

    for entry_id, group in groups.items():
        labels = sorted(ex.gold_label for ex in group)
        
        if labels == ["accurate", "mischaracterized"]:
            complete[entry_id] = group
        else:
            incomplete += 1
            log.warning("incomplete_pair", entry_id=entry_id, labels=labels)

    if incomplete:
        log.warning("incomplete_pairs_total", count=incomplete)

    return complete


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def _resolve_entry_case(
    citation_text: str,
    client: CourtListenerClient,
    cache: Cache,
) -> ResolvedCase | None:
    """
    Resolve a citation string to its full opinion text.

    Parses the citation text with eyecite (the same parser used during
    set construction, so this should always succeed) and resolves it
    via CourtListener with caching.

    Args:
        citation_text: The citation string from the CaseHOLD example.
        client: CourtListener API client.
        cache: Pipeline cache instance.

    Returns:
        The resolved case, or None if parsing or resolution failed.
    """
    citations = parse_citations(citation_text)
    if not citations:
        log.warning("citation_reparse_failed", citation=citation_text)

        return None

    resolved = resolve_citation(citations[0], client=client, cache=cache)
    if isinstance(resolved, Abstention):
        log.info(
            "resolution_abstained",
            citation=citation_text,
            reason=resolved.reason,
        )

        return None

    return resolved


# ---------------------------------------------------------------------------
# Record construction
# ---------------------------------------------------------------------------


def _build_record(
    example: EvalExample,
    case: ResolvedCase,
    embedding_client: EmbeddingClient,
    cache: Cache,
    *,
    chunk_max_tokens: int,
    chunk_overlap_paragraphs: int,
    top_k: int,
) -> dict | None:
    """
    Build a single prompt/completion training record.

    Runs the retrieval stage (chunk → embed → top-K) for the example's
    claim against the resolved opinion, then assembles the classification
    prompt and a label-only completion.

    Args:
        example: The CaseHOLD-derived example (accurate or mischaracterized).
        case: The resolved cited case with full opinion text.
        embedding_client: Voyage embedding client.
        cache: Pipeline cache instance.
        chunk_max_tokens: Maximum approximate tokens per chunk.
        chunk_overlap_paragraphs: Paragraph overlap between chunks.
        top_k: Number of chunks to retrieve.

    Returns:
        A dict with ``prompt``, ``completion``, and traceability fields,
        or None if the opinion produced no chunks.
    """
    claim = example.metadata.get("claim")
    if not claim:
        log.warning("missing_claim", example_id=example.example_id)

        return None

    chunks = chunk_opinion(
        text=case.full_text,
        max_tokens=chunk_max_tokens,
        overlap_paragraphs=chunk_overlap_paragraphs,
    )
    if not chunks:
        log.warning("no_chunks", example_id=example.example_id)

        return None

    chunk_embeddings = embed_chunks(chunks, client=embedding_client, cache=cache)
    claim_embedding = embed_claim(claim, client=embedding_client, cache=cache)
    retrieval = retrieve_top_k(claim_embedding, chunk_embeddings, k=top_k)

    # Format retrieved chunks the same way the classify stage does.
    retrieved_text = "\n\n---\n\n".join(
        f"[Excerpt {i + 1}]\n{chunk.text}" for i, chunk in enumerate(retrieval.chunks)
    )

    prompt = build_classification_prompt(
        claim=claim,
        retrieved_text=retrieved_text,
        case_name=case.case_name,
    )

    # Label-only completion: the fine-tuned model learns to emit just the
    # binary verdict. The classify stage applies defaults for the optional
    # confidence/supporting_text fields at inference time.
    completion = json.dumps({"label": example.gold_label})

    return {
        "prompt": prompt,
        "completion": completion,
        # Traceability fields — ignored by the training scripts.
        "example_id": example.example_id,
        "label": example.gold_label,
        "split": example.split,
        "citation_text": example.citation_text,
        "case_name": case.case_name,
        "top_retrieval_score": max(retrieval.scores),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build prompt/completion fine-tuning data from CaseHOLD "
            "accurate/mischaracterized pairs."
        ),
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to the CaseHOLD JSONL file (output of build_casehold_set.py).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/training"),
        help="Directory for train.jsonl and val.jsonl (default: data/training).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to configuration file. Default: config.yaml.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proportion of entries assigned to val (default: 0.15). "
        "The rest go to train; the real-brief set is the held-out test set.",
    )
    parser.add_argument(
        "--min-retrieval-score",
        type=float,
        default=0.0,
        help="Drop a pair if its ACCURATE example's best retrieval score "
        "is below this threshold (label-noise guard). Default: 0.0 (off).",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Process at most this many entries. Useful for smoke tests.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split assignment (default: 42).",
    )
    args = parser.parse_args()

    # ---- Config, secrets, clients ----

    config = load_config(args.config)
    configure_logging(level=config.log_level)

    load_dotenv()
    secrets = load_secrets()

    cache = Cache(path=config.cache_dir, enabled=True)

    courtlistener_client = CourtListenerClient(
        api_key=secrets.courtlistener_api_key,
        base_url=config.courtlistener_base_url,
        rate_limit_per_minute=config.courtlistener_rate_limit_per_minute,
        max_retries=config.courtlistener_max_retries,
        timeout_seconds=config.llm_timeout_seconds,
    )

    embedding_client = EmbeddingClient(
        api_key=secrets.voyage_api_key,
        model=config.embedding_model,
    )

    # ---- Load examples and assign splits ----

    examples = validate_labels(load_casehold(args.source))
    if not examples:
        print(f"No valid examples found in {args.source}", file=sys.stderr)
        sys.exit(1)

    # Train/val only — the real-brief set is the held-out test set, so
    # the test ratio is the leftover after train + val, which is ~0.
    assign_splits(
        examples,
        train_ratio=1.0 - args.val_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    pairs = _group_into_pairs(examples)
    entry_ids = sorted(pairs.keys())

    if args.max_entries:
        entry_ids = entry_ids[: args.max_entries]

    print(f"Loaded {len(examples)} examples ({len(pairs)} complete pairs).")
    print(f"Processing {len(entry_ids)} entries...")

    # ---- Process entries ----

    records = []
    drop_reasons: Counter[str] = Counter()
    accurate_scores = []

    for i, entry_id in enumerate(entry_ids):
        pair = pairs[entry_id]

        # Resolve once per entry — both pair members cite the same case.
        case = _resolve_entry_case(pair[0].citation_text, courtlistener_client, cache)
        if case is None:
            drop_reasons["resolution_failed"] += 1
            continue

        pair_records = []
        for example in pair:
            record = _build_record(
                example,
                case,
                embedding_client,
                cache,
                chunk_max_tokens=config.chunk_max_tokens,
                chunk_overlap_paragraphs=config.chunk_overlap_paragraphs,
                top_k=config.top_k,
            )
            if record is None:
                break

            pair_records.append(record)

        if len(pair_records) != 2:
            drop_reasons["record_build_failed"] += 1
            continue

        # Label-noise guard: if the accurate example's claim doesn't match
        # anything in the retrieved excerpts, drop the whole pair.
        accurate_record = next(r for r in pair_records if r["label"] == "accurate")
        accurate_scores.append(accurate_record["top_retrieval_score"])

        if accurate_record["top_retrieval_score"] < args.min_retrieval_score:
            drop_reasons["low_retrieval_score"] += 1
            continue

        records.extend(pair_records)

        if (i + 1) % 100 == 0:
            print(
                f"  [{i + 1}/{len(entry_ids)}] "
                f"{len(records)} records built, {sum(drop_reasons.values())} entries dropped"
            )

    courtlistener_client.close()
    cache.close()

    # ---- Report ----

    print(f"\nDone: {len(records)} records from {len(records) // 2} entries.")
    if drop_reasons:
        print("Dropped entries:")
        for reason, count in drop_reasons.most_common():
            print(f"  {reason}: {count}")

    if accurate_scores:
        sorted_scores = sorted(accurate_scores)

        def pct(p: float) -> float:
            return sorted_scores[min(int(p * len(sorted_scores)), len(sorted_scores) - 1)]

        print(
            "Accurate-example retrieval scores: "
            f"min={sorted_scores[0]:.3f}, p10={pct(0.10):.3f}, "
            f"p50={pct(0.50):.3f}, p90={pct(0.90):.3f}, max={sorted_scores[-1]:.3f}"
        )
        print(
            "If the low tail looks weak, re-run with --min-retrieval-score "
            "(cached resolutions/embeddings make re-runs fast)."
        )

    # ---- Write output ----

    args.output_dir.mkdir(parents=True, exist_ok=True)

    by_split: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_split[record["split"]].append(record)

    for split_name in ("train", "val"):
        out_path = args.output_dir / f"{split_name}.jsonl"
        split_records = by_split.get(split_name, [])

        with open(out_path, "w", encoding="utf-8") as f:
            for record in split_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        label_counts = Counter(r["label"] for r in split_records)
        print(
            f"Wrote {len(split_records)} records to {out_path} "
            f"({dict(label_counts)})"
        )

    # Anything that landed in "test" (shouldn't happen with val-only split
    # config, but possible due to rounding) is reported, not silently lost.
    if by_split.get("test"):
        print(
            f"Note: {len(by_split['test'])} records were assigned to 'test' "
            "by split rounding and were NOT written. Adjust --val-ratio if "
            "this matters."
        )


if __name__ == "__main__":
    main()

# Evaluation Summary

**Run ID:** 9c1b40f8-46c0-4aec-9b62-1eac26fa8b28
**Timestamp:** 2026-06-27T17:55:09.908678+00:00
**Git SHA:** bf238ee
**Models evaluated:** gemma12b-prompted-modal, gemma12b-tuned
**Sources:** real_brief

## Macro F1 by Source and Model

| Source | gemma12b-prompted-modal | gemma12b-tuned |
|---|---|---|
| real_brief | 0.6928 | 0.8294 |

## real_brief

### gemma12b-prompted-modal

- **Macro F1:** 0.6928
- **Examples:** 189
- **Abstention rate:** 25.40%
- **Abstentions by reason:** case-not-found: 35, parsing-failed: 13

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| accurate | 0.9394 | 0.4493 | 0.6078 | 69 |
| mischaracterized | 0.6481 | 0.9722 | 0.7778 | 72 |

### gemma12b-tuned

- **Macro F1:** 0.8294
- **Examples:** 189
- **Abstention rate:** 25.40%
- **Abstentions by reason:** case-not-found: 35, parsing-failed: 13

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| accurate | 0.8462 | 0.7971 | 0.8209 | 69 |
| mischaracterized | 0.8158 | 0.8611 | 0.8378 | 72 |

---

Full metrics available in `metrics.json`. Reproducibility record in `run_manifest.json`.
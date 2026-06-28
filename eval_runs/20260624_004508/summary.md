# Evaluation Summary

**Run ID:** cfb59a2c-0512-40f2-94c8-dfda2a2ec7fc
**Timestamp:** 2026-06-24T00:45:08.701134+00:00
**Git SHA:** bf238ee
**Models evaluated:** gemma12b-prompted-modal, gemma12b-tuned
**Sources:** real_brief

## Macro F1 by Source and Model

| Source | gemma12b-prompted-modal | gemma12b-tuned |
|---|---|---|
| real_brief | 0.5000 | 0.5000 |

## real_brief

### gemma12b-prompted-modal

- **Macro F1:** 0.5000
- **Examples:** 10
- **Abstention rate:** 60.00%
- **Abstentions by reason:** case-not-found: 3, parsing-failed: 3

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| accurate | 0.0000 | 0.0000 | 0.0000 | 0 |
| mischaracterized | 1.0000 | 1.0000 | 1.0000 | 4 |

### gemma12b-tuned

- **Macro F1:** 0.5000
- **Examples:** 10
- **Abstention rate:** 60.00%
- **Abstentions by reason:** case-not-found: 3, parsing-failed: 3

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| accurate | 0.0000 | 0.0000 | 0.0000 | 0 |
| mischaracterized | 1.0000 | 1.0000 | 1.0000 | 4 |

---

Full metrics available in `metrics.json`. Reproducibility record in `run_manifest.json`.
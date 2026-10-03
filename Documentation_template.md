# ML Challenge 2026: Business Entity Resolution

**Team Name:** Samadhan, MNIT Jaipur
**Team Members:** < Sakshi Singhal, Somya Vijay, Uttam Rathore, Priyanshu Mundotia>
**Submission Date:** October 2, 2026

---

## 1. Executive Summary

We match Source-1 (S1) business entities to Source-2/Source-3 (S2/S3) records in two stages. The first is country-aware retrieval (blocking), which produces at most 10 candidate S1 entities for every S2/S3 record. The second is a LightGBM matching model: a pairwise C3 model, followed by a per-record C4 model that re-scores each record's top candidates. The recommended final output keeps the C4 decisions for the US and India. For France, which is absent from the training data, it adds France-specific normalization and a pseudo-label model with a threshold of 0.9.

## 2. Methodology

### 2.1 Problem Analysis

Records differ in business-name spelling, legal suffixes, transliteration (including Indic scripts), address formatting and missing address fields. The test set holds about 1.73M S1 entities and about 9.97M S2/S3 records, so exhaustive pairwise comparison is not feasible. France has no training data, so it needs separate handling.

### 2.2 Solution Strategy

1. **Normalization.** Names and addresses are normalized. An Indic token dictionary, learned only from TRAIN-fold pairs, maps Indic-script names. France-only French normalization covers legal forms, regions/departments and address tokens.
2. **Blocking.** Country-aware retrieval builds one S1 index per country and queries it with S2/S3 records (see Section 3).
3. **Pairwise scoring (C3).** LightGBM scores every candidate pair.
4. **Per-record scoring (C4).** A second LightGBM stage uses each record's top-candidate probabilities and margin.
5. **France adaptation.** France-only normalization and one round of pseudo-label self-training.
6. **Assembly.** Matches must come from the generated candidate set, and each S2/S3 record is assigned to at most one S1 entity.

**Approach Type:** Retrieval blocking + gradient-boosted matching model
**Core Innovation:** Direct pair evidence plus per-record second-stage scoring, with targeted French normalization and pseudo-labeling for the unseen test country.

## 3. Candidate Generation (Blocking)

- **Blocking method:** Per-country S1 index. Names use character 3-gram TF-IDF and addresses use word-level TF-IDF. Both are compressed with CountSketch (1024 dimensions, seeds 42/43). Retrieval is approximate top-40, followed by exact sparse-cosine rescoring that keeps the top 10 candidates per S2/S3 record.
- **Candidate pairs generated:** 99,695,890 pairs in `output/candidate_pairs.tsv`, covering all 1,732,544 S1 entities.
- **How true matches were preserved:** Development-data audits reported 98.7% candidate recall. Every one of the 5,761,834 matched IDs in the final output appears in the candidate file.
- **Regeneration disclosure:** The original candidate file from the v4 run was lost. `candidate_pairs.tsv` was regenerated with the same code and retrieval settings, on different hardware (RTX 3050 4 GB, CPU/GPU with 50,000-record query chunks instead of 200,000). Approximate top-k can differ slightly across hardware, so the file is not byte-identical to the original (original SHA-256: `dc13405de5a94849b97c9188dade5a6975269634c30185206ed6f7b4ede33688`). A France spot check on 1,000 queries recovered 99.68% of the historical matches within the top 10 (622 of 624).

## 4. Matching Model

**Features used:**
- Name similarity, token agreement, raw-text agreement and direct evidence.
- Address similarity, address-number agreement, number containment/closeness and rare-token evidence.
- Candidate rank, first-stage (C3) probability and per-record top-candidate margin (C4).

**Model type:** LightGBM binary classifier in both stages (C3 pairwise, C4 per-record).
**Threshold selection method:** Thresholds were chosen on validation splits. The C4 baseline uses 0.7. The recommended France-adjusted variant uses 0.9 for France, while the US and India stay at 0.7.

## 5. Results & Error Analysis

- **C4 baseline validation F0.5 (macro):** 0.98322 on DEV and 0.98321 on CONF.
- **C4 baseline leaderboard score:** 0.970441.
- **France-adjusted output:** Its leaderboard score has not been measured, and the France test set has no labels. The DEV/CONF figures above are local validation results and are not a score for the France-adjusted variant.
- **Common false positives:** Similar names at different addresses, especially among France pseudo-label decisions.
- **Common false negatives:** True matches with sparse or ambiguous names and addresses, which retrieval may miss or the final threshold may reject.

## 6. Conclusion

Country-aware retrieval keeps the search space small, and the two-stage LightGBM model gives strong validation scores (F0.5 about 0.983). A separate French normalization and pseudo-labeling path targets the unseen country. The C4 baseline's leaderboard result is known, but the France-adjusted variant remains unmeasured on the leaderboard. The candidate file was regenerated, with the limitation described in Section 3.

## Appendix

### A. Code Artefacts

Runnable source is under `code/business_entity_resolution/src/`. Base pipeline code is in `src/`, C3/C4 code in `src/v2/`, and France-specific code in `src/v4/`. Reproduction steps, memory requirements and runtime notes are in `code/business_entity_resolution/README.md`.

### B. Additional Results

- **Matching output:** 1,732,544 S1 rows, 1,633,230 with at least one match, 5,761,834 matched IDs.
- **Candidate output:** 1,732,544 S1 rows, 99,695,890 candidate IDs.
- **Regeneration environment:** Python 3.12, CUDA PyTorch, RTX 3050 (4 GB VRAM), 16 GB RAM. Peak VRAM per country was 1.25 GiB (France), 2.02 GiB (US) and 2.30 GiB (India).
- **Code changes for memory only:** chunked normalization (500,000 rows), per-country candidate runs with 50,000-record query chunks, and a CPU fallback on GPU out-of-memory. One further change, a fallback to `name_norm` when `name_norm_raw` is absent, could affect Indic-name normalization.
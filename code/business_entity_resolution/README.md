# Business Entity Resolution: Reproduction Guide

## 1. What this does

The pipeline matches each Source-1 (S1) business to its records in Source-2/Source-3 (S2/S3) in two stages:

1. **Blocking.** Country-aware retrieval (character 3-gram name TF-IDF, word TF-IDF addresses, CountSketch 1024 dims, seeds 42/43, approximate top-40, exact rescoring) keeps the top 10 candidate S1 entities per S2/S3 record. The result is `candidate_pairs.tsv`.
2. **Matching.** LightGBM C3 (pairwise) and C4 (per-record) score the candidates, and a threshold produces `matching_results.tsv`. The recommended final variant (v4) adds France-specific normalization and a pseudo-label model (France threshold 0.9; US/India 0.7).

## 2. Folder layout

Raw TSV Files
     ↓
TSV → Parquet
     ↓
Normalization
     ↓
Indic Dictionary
     ↓
Candidate Generation
     ↓
Feature Extraction
     ↓
Model Scoring
     ↓
Thresholding
     ↓
Final Matching
     ↓
candidate_pairs.tsv
matching_results.tsv


## 3. Environment

- Python 3.12, Windows/PowerShell. The commands below are PowerShell.
- `pip install -r requirements.txt` (versions pinned; it installs the CUDA 12.1 PyTorch wheel).
- A GPU is optional. Without CUDA, retrieval runs on CPU, only slower.
- Tested on 16 GB RAM and an RTX 3050 (4 GB VRAM).

**Measured resource use (regeneration run):**

| Stage | Peak |
|---|---|
| `build_normalized.py` | about 5 GiB process RAM (full tables) |
| Candidate generation VRAM | France 1.25 GiB, US 2.02 GiB, India 2.30 GiB |

## 4. Input data

Set `ER_DATA_DIR` to the dataset folder:

- `train\train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, `train_ground_truth.tsv`
- `test\test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv`

Each source TSV has `entity_id`, `business_name`, `business_address`, `country`. The ground truth has `source1_entity_id` and `matched_entity_ids`.

**Metric.** Per S1 entity, precision = TP / predicted matches and recall = TP / ground-truth matches. The score is F0.5 = 1.25·P·R / (0.25·P + R), macro-averaged over evaluated S1 entities, including singletons.

## 5. Run the pipeline

Run from the submission root.

```powershell
$submission = (Get-Location).Path
$env:ER_DATA_DIR = 'C:\path\to\dataset'
$env:ER_CACHE_DIR = Join-Path $submission 'work\cache'
$env:ER_OUTPUT_DIR = Join-Path $submission 'work\base_output'
$env:ER_FINAL_OUTPUT_DIR = Join-Path $submission 'output'
New-Item -ItemType Directory -Force -Path $env:ER_CACHE_DIR, $env:ER_OUTPUT_DIR | Out-Null
Set-Location (Join-Path $submission 'code\business_entity_resolution')
python -m pip install -r requirements.txt
$env:PYTHONPATH = Join-Path (Get-Location).Path 'src'

# Step 1: base preparation, normalization, candidates/features, training, scoring
python src\run_all.py

# Step 2: Indic dictionary, C3/C4 training and scoring, bundle (threshold 0.7)
python src\v2\run_v2.py 0.7

# Step 3: France v4 path (needs the v2 artifacts and models from Step 2)
python src\v4\v4_prepare.py
python src\candidates.py v4test --country France
python -c "import build_features; build_features.build('v4test')"
python src\v2\featx.py v4test
python src\v2\train_v2.py C3_direct_x_more_data v4test
python src\v2\stack_v2.py C3_direct_x_more_data v2train v4test
python src\v4\fr_pseudo.py fr1 0.98 0.02 0.25 0.10
python src\v4\export_candidates.py
python src\v4\assemble.py c4fr_frp_t90 frp_fr1 0.9
```

| Step | Reads | Writes |
|---|---|---|
| 1 `run_all.py` | the six source TSVs and train ground truth | parquet cache; base submission in `ER_OUTPUT_DIR` |
| 2 `run_v2.py` | Step 1 cache | `work\cache\runs\submission_C4_stack_on_C3_direct_x_more_data\` (`matching_results.tsv`, `candidate_pairs.tsv`) |
| 3 v4 path | v2 test artifacts and models | final `matching_results.tsv` and `candidate_pairs.tsv` in `output\` |

Completed stages are skipped on re-run (resumable). `v4_prepare.py` needs the `v2test_*_norm.parquet` files from Step 2, so run Step 2 first.

## 6. Low-memory options

- `src\build_normalized.py` accepts `--split`, `--source` and `--chunk-rows`. It normalizes in chunks, validates row count and `entity_id` order, and writes through a temporary file. Example: `python src\build_normalized.py --split test --source source2 --chunk-rows 500000`.
- `src\candidates.py` runs one country at a time, with 50,000-record query chunks. It skips countries that are already done, and falls back to CPU if the GPU runs out of memory.
- Close other applications before the normalization and candidate stages.

## 7. Validate

```powershell
python src\utils\validate_submission.py --matching "$env:ER_FINAL_OUTPUT_DIR\matching_results.tsv" --candidate "$env:ER_FINAL_OUTPUT_DIR\candidate_pairs.tsv" --test-dir "$env:ER_DATA_DIR\test"
```

The validator checks the format, S1 coverage and candidate containment. ID-existence checks are optional (`--check-ids`) and use more memory. Both files in `output\` are the submission files.

## 8. Notes and limitations

- **Regenerated candidates.** The original `candidate_pairs.tsv` was lost, so the file in `output/` was regenerated with the same code and retrieval settings on different hardware. Approximate top-k can differ slightly across hardware, so it is not byte-identical to the original (original SHA-256 `dc13405de5a94849b97c9188dade5a6975269634c30185206ed6f7b4ede33688`). All 5,761,834 matched IDs in `matching_results.tsv` appear in it.
- **Code changes.** The changes made during regeneration are chunked normalization, per-country candidate runs with 50,000-record chunks and CPU fallback, and a `name_norm` fallback in `v2/v2_prepare.py` when `name_norm_raw` is absent. The last one could slightly affect Indic-name normalization.
- **France.** France has no training data and no test labels, so its threshold (0.9) and pseudo-label model are validated only indirectly.
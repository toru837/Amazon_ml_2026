# Business Entity Resolution — Amazon ML Challenge 2026

Match each business in **Source 1 (S1)** to its records in **Source 2 (S2)** and **Source 3 (S3)**, across noisy names and addresses in multiple countries and scripts (US, France, India, including Indic-script names).

> **Leaderboard rank:** `1776/27000`  |  **Team:** `TEAM SAMADHAN`

---

## 1. Problem

Three business directories describe the same real-world companies, but with different spellings, abbreviations, address formats, languages and scripts. For every S1 entity, predict which S2/S3 records refer to the same business.

- **Inputs:** `source1/2/3.tsv`, each with `entity_id`, `business_name`, `business_address`, `country`.
- **Ground truth (train only):** `source1_entity_id`, `matched_entity_ids`.
- **Output:** `matching_results.tsv` (`source1_entity_id`, `matched_entity_ids`) plus `candidate_pairs.tsv`.
- **Metric:** per S1 entity, precision = TP / predicted matches, recall = TP / true matches. Score is **F0.5** (precision weighted more than recall), macro-averaged over S1 entities, singletons included.

```
F0.5 = 1.25 · P · R / (0.25 · P + R)
```

---

## 2. Approach

A two-stage pipeline: cheap **blocking** to shrink the search space, then a learned **matcher** to decide.

```
Raw TSV files
   ↓  prepare_data.py        TSV → Parquet
   ↓  normalization          lowercase, strip punctuation, unify address tokens
   ↓  Indic dictionary       map Indic-script names to Latin tokens
   ↓  Candidate generation   country-aware TF-IDF retrieval (top-10 per record)
   ↓  Feature extraction     name/address similarity features
   ↓  Model scoring          LightGBM C3 (pairwise) + C4 (per-record stack)
   ↓  Thresholding           per-country thresholds
   ↓
candidate_pairs.tsv  →  matching_results.tsv
```

### Stage 1 — Blocking (candidate generation)
- Country-aware: records only compete within their own country.
- Character 3-gram TF-IDF on names, word TF-IDF on addresses.
- CountSketch projection to 1024 dimensions (seeds 42/43), approximate top-40, then **exact rescoring**.
- Keeps the **top 10** candidate S1 entities per S2/S3 record → `candidate_pairs.tsv`.

### Stage 2 — Matching
- **C3 (pairwise):** LightGBM scores each (record, candidate) pair from similarity features (name cosine, address cosine, combined cosine, rank, candidate count, score gaps, fuzzy ratios via RapidFuzz).
- **C4 (per-record stack):** a second LightGBM stacked on C3 using per-record context (rank, gap to next candidate, number of candidates).
- **Thresholding:** accept a pair when the score passes the country threshold.

### Final variant (v4)
Adds **France-specific normalization** and a **pseudo-label model**. Thresholds: **France 0.9**, **US/India 0.7**.

### Key design choices
- Blocking by country keeps retrieval fast and avoids cross-country false positives.
- Exact rescoring after approximate retrieval keeps recall high at low memory.
- F0.5 favours precision, so thresholds are set conservatively, especially for France.
- Memory-guarded normalization (RAM guard) so it runs on a 16 GB laptop.

---

## 3. Repository layout

```
.
├── Documentation_template.md
└── code/business_entity_resolution/
    ├── README.md
    ├── requirements.txt
    ├── business_entity_sample_100rows.zip   # 100-row sample for quick testing
    └── src/
        ├── config.py              # paths via ER_* environment variables
        ├── prepare_data.py        # TSV → Parquet
        ├── normalization.py       # text/address normalization
        ├── build_normalized.py    # normalize all splits
        ├── indic_dictionary.py    # build Indic → Latin dictionary
        ├── apply_indic_dict.py    # apply dictionary
        ├── retrieval.py           # TF-IDF + CountSketch country index
        ├── candidates.py          # candidate generation
        ├── features.py            # pair features
        ├── build_features.py      # feature table builder
        ├── train.py               # LightGBM training
        ├── predict.py             # scoring + thresholding
        ├── metrics.py             # F0.5 evaluation
        ├── run_all.py             # base pipeline entry point
        ├── utils/                 # helpers, submission validator
        ├── v2/                    # Indic dictionary + C3/C4 + bundling
        └── v4/                    # France normalization + pseudo-label model
```

---

## 4. Setup

**Requirements:** Python 3.12 recommended, 16 GB RAM for the full dataset. A GPU is optional (retrieval falls back to CPU, slower).

```bash
git clone https://github.com/toru837/Amazon_ml_2026.git
cd Amazon_ml_2026/code/business_entity_resolution
python3 -m venv .venv && source .venv/bin/activate     # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If the pinned CUDA torch wheel (`torch==2.5.1+cu121`) is unavailable on your machine, install a matching torch first and then the rest:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
grep -vi "^torch" requirements.txt > req_notorch.txt && pip install -r req_notorch.txt
```

On Linux, LightGBM also needs `sudo apt install -y libgomp1`.

---

## 5. Dataset and results (not included in this repository)

The complete dataset and the final result files (`candidate_pairs.tsv`, `matching_results.tsv`) are **not uploaded to this repository** because of GitHub's file size limit. They are provided separately and should be placed as described below.

- **Full dataset:** download it from the link provided separately and extract it to a local folder, then point `ER_DATA_DIR` at that folder.
- **Final results:** the generated `candidate_pairs.tsv` and `matching_results.tsv` are shared separately. You can also regenerate them by running the pipeline on the full dataset.
- **Included here:** a 100-row sample (`business_entity_sample_100rows.zip`) so the pipeline can be tested quickly without the full data.

> Dataset link: `<paste link here>`
> Results link: `<paste link here>`

### Input data layout

Set `ER_DATA_DIR` to a folder containing:

```
train/  train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
test/   test_source1.tsv   test_source2.tsv   test_source3.tsv
```

---

## 6. Run

### Linux / macOS / EC2 (bash)
```bash
export ER_DATA_DIR=/path/to/dataset
export ER_CACHE_DIR=$PWD/work/cache
export ER_OUTPUT_DIR=$PWD/work/base_output
export ER_FINAL_OUTPUT_DIR=$PWD/output
mkdir -p $ER_CACHE_DIR $ER_OUTPUT_DIR $ER_FINAL_OUTPUT_DIR
export PYTHONPATH=$PWD/src

python src/run_all.py            # Step 1: prepare, normalize, candidates, features, train, score
python src/v2/run_v2.py 0.7      # Step 2: Indic dictionary, C3/C4, bundle (threshold 0.7)
# Step 3: France v4 path (see src/v4 and the original reproduction guide)
```

### Windows (PowerShell)
```powershell
$env:ER_DATA_DIR = 'C:\path\to\dataset'
$env:ER_CACHE_DIR = "$PWD\work\cache"
$env:ER_OUTPUT_DIR = "$PWD\work\base_output"
$env:ER_FINAL_OUTPUT_DIR = "$PWD\output"
New-Item -ItemType Directory -Force -Path $env:ER_CACHE_DIR, $env:ER_OUTPUT_DIR, $env:ER_FINAL_OUTPUT_DIR | Out-Null
$env:PYTHONPATH = "$PWD\src"

python src\run_all.py
python src\v2\run_v2.py 0.7
```

The final submission files are written under `work/cache/runs/submission_C4_stack_on_C3_direct_x_more_data/`; copy them to `output/`.

### Quick test on the 100-row sample
```bash
unzip business_entity_sample_100rows.zip -d sample
export ER_DATA_DIR=$PWD/sample/business_entity_sample
```
Then run the commands above. **Sample-only note:** on 100 rows the full-data TF-IDF `min_df`/`max_df` values prune every term. For the sample run only, relax them in `src/retrieval.py` (`min_df=1`, `max_df=1.0`), and restore the originals for real data. Metrics on the sample (perfect fit, accept rate 1.0) are not meaningful; it only proves the pipeline runs end to end.

---

## 7. Resource usage (full data)

| Stage | Peak |
|---|---|
| `build_normalized.py` | about 5 GiB RAM |
| Candidate generation VRAM | France 1.25 GiB, US 2.02 GiB, India 2.30 GiB |
| Submission validator | about 10 GB RAM |

Tested on a laptop with 16 GB RAM and an RTX 3050 (4 GB VRAM). On a cloud VM use 16 GB RAM or more and 40 GB disk or more.

---

## 8. Results

\[
F_{0.5}=\frac{1.25PR}{0.25P+R}
\]

| Variant | Description | Precision | Recall | F0.5 |
|---|---|---:|---:|---:|
| Earlier prototype | Sampled corpus | 0.8500 | 0.9850 | **0.8757** |
| v1 | TF-IDF blocking + LightGBM | 0.9407 | 0.9735 | **0.9468** |
| v2 | + Indic dictionary, C3/C4 stack (threshold 0.7) | 0.9830 | 0.9890 | **0.9842** |

**Leaderboard:** rank `1776` for `Team Samadhan`.

---

## 9. What worked well
- Country-aware blocking: fast, high recall, low memory.
- Exact rescoring over approximate top-40 candidates.
- Indic-script dictionary lifted matching for Indian records.
- Per-country thresholds tuned for F0.5 (precision-leaning).
- Stacked per-record model (C4) using rank and score gaps.

## 10. Limitations and future work
- Thresholds are hand-tuned per country; could be learned.
- Pseudo-labels may reinforce model errors.
- Dense multilingual embeddings could replace or augment TF-IDF blocking.
- Add automated tests and a single cross-platform run script.

---

## 11. Author

**UTTAM RATHORE** — B.Tech ECE, MNIT Jaipur.

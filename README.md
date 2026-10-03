# Business Entity Resolution --- Amazon ML Challenge 2026

Match each business in Source 1 (S1) with the same business in Source 2
(S2) and Source 3 (S3).

The data is noisy: names, addresses, spellings, countries, and writing
scripts can be different.

**Leaderboard:** 1776/27000\
**Team:** TEAM SAMADHAN

## 1. Problem

Three business directories can describe the same company in different
ways.

Example:

``` text
S1: Sharma Electronics Pvt Ltd
S2: Sharma Electronics Private Limited
S3: शर्मा इलेक्ट्रॉनिक्स
```

The goal is to find which S2/S3 records belong to each S1 business.

Each source contains:

``` text
entity_id
business_name
business_address
country
```

Training data also contains:

``` text
source1_entity_id
matched_entity_ids
```

The final files are:

``` text
candidate_pairs.tsv
matching_results.tsv
```

The competition uses F0.5, which gives more importance to precision.

``` text
Precision = correct predictions / all predictions
Recall    = correct predictions / all true matches
F0.5      = 1.25 × P × R / (0.25 × P + R)
```

The score is macro-averaged over S1 entities, including singletons.

## 2. Pipeline --- Simple Explanation

The pipeline has two main jobs:

``` text
Find possible matches
        ↓
Choose the real matches
```

### Step 1 --- Prepare the data

**TSV files are converted into Parquet files, making large data faster
and easier to process.**

``` text
TSV → Parquet
```

Uses column-wise chunks to reduce memory usage.

File: `src/prepare_data.py`

### Step 2 --- Normalize names and addresses

**We clean names and addresses so small writing differences do not
confuse the matching system.**

Examples:

``` text
"Pvt. Ltd."    → "private limited"
"ABC-STORE"    → "abc store"
```

Files: `src/normalization.py`, `src/build_normalized.py`

### Step 3 --- Handle Indian-language names

**Indian-script names are converted into useful Latin tokens so similar
businesses can be compared.**

Example:

``` text
शर्मा इलेक्ट्रॉनिक्स
        ↓
Sharma Electronics
```

Files: `src/indic_dictionary.py`, `src/apply_indic_dict.py`

### Step 4 --- Split by country

**Businesses are compared only with businesses from the same country,
avoiding many impossible matches.**

``` text
India → India
US → US
France → France
```

### Step 5 --- Find possible candidates

**We find similar S1 businesses instead of comparing every business with
every other business.**

This is called **blocking**.

``` text
Millions of records
       ↓
Find likely records
       ↓
Keep only candidates
```

### Step 6 --- Compare names using TF-IDF

**Character 3-gram TF-IDF finds names that look similar even when
spelling is slightly different.**

Example:

``` text
Enterprises
Enterprizes
```

Their character pieces are similar, so they become possible candidates.

### Step 7 --- Compare addresses using TF-IDF

**Word-level TF-IDF compares address words and helps identify businesses
with similar locations.**

Example:

``` text
12 MG Road Jaipur
12, MG Rd, Jaipur
```

### Step 8 --- Fast candidate retrieval

**CountSketch quickly retrieves approximately similar records before
exact similarity checking.**

Technical settings:

``` text
CountSketch: 1024 dimensions
Seeds: 42 / 43
Approximate top-40
```

File: `src/retrieval.py`

### Step 9 --- Exact candidate rescoring

**The approximate candidates are checked again using exact similarity to
keep the strongest candidates.**

``` text
Approximate top-40
        ↓
Exact rescoring
        ↓
Top 10 S1 candidates
```

The result is `candidate_pairs.tsv`.

File: `src/candidates.py`

## 3. Matching Stage

### Step 10 --- Create matching features

**We calculate many clues showing how similar two business records
are.**

Important features:

``` text
Name cosine similarity
Address cosine similarity
Combined cosine similarity
Name fuzzy similarity
Address fuzzy similarity
Candidate rank
Candidate count
Score gaps
```

RapidFuzz is used for fuzzy comparisons.

Files: `src/features.py`, `src/build_features.py`

### Step 11 --- LightGBM C3 model

**LightGBM learns from these clues and predicts how likely each
candidate pair is a real match.**

Example:

``` text
Name similarity    → High
Address similarity → High
Candidate rank     → 1

             ↓

       Strong match
```

C3 is the pairwise matching model.

File: `src/train.py`

### Step 12 --- C4 per-record model

**C4 looks at each record's candidates together, using ranks and score
gaps to improve decisions.**

Example:

``` text
Candidate A → 0.92
Candidate B → 0.51
Candidate C → 0.43
```

The large score gap gives useful context.

C4 is a second LightGBM model stacked on C3.

### Step 13 --- Apply thresholds

**A candidate is accepted only when its predicted probability crosses
the selected country threshold.**

Main v2 thresholds:

``` text
US    → 0.7
India → 0.7
```

Final France threshold:

``` text
France → 0.9
```

### Step 14 --- Final matching

**Accepted candidates are collected for every S1 business and written
into the final matching file.**

Final file:

``` text
matching_results.tsv
```

## 4. Final v4 Improvements

### France normalization

**France-specific text cleaning helps the model handle French business
names and addresses more consistently.**

### France pseudo-label model

**Pseudo-labeling creates additional useful training examples because
France has no direct training labels.**

Final thresholds:

``` text
France → 0.9
US     → 0.7
India  → 0.7
```

Files: `src/v4/`

## 5. Complete Pipeline

``` text
Raw TSV files
      ↓
Prepare data
      ↓
TSV → Parquet
      ↓
Normalization
      ↓
Indic dictionary
      ↓
Country-aware blocking
      ↓
TF-IDF retrieval
      ↓
CountSketch
      ↓
Approximate top-40
      ↓
Exact rescoring
      ↓
Top-10 candidates
      ↓
candidate_pairs.tsv
      ↓
Feature extraction
      ↓
LightGBM C3
      ↓
C4 per-record stack
      ↓
Country thresholds
      ↓
Final matching
      ↓
matching_results.tsv
```

## 6. Key Design Choices

-   **Country-aware blocking:** Comparing within countries makes
    retrieval faster and prevents many cross-country false matches.
-   **Character 3-gram TF-IDF:** Small character pieces handle spelling
    mistakes and slightly different business names well.
-   **Exact rescoring:** Approximate retrieval is fast, while exact
    rescoring improves selected candidate quality.
-   **Indic dictionary:** Latin representations of Indian-script names
    help matching across different writing systems.
-   **F0.5 thresholds:** F0.5 rewards precision more, so thresholds are
    chosen carefully to avoid incorrect matches.
-   **C3 + C4:** C3 scores pairs, while C4 also considers how each
    candidate compares with competitors.

## 7. Repository Layout

``` text
.
├── Documentation_template.md
└── code/
    └── business_entity_resolution/
        ├── README.md
        ├── requirements.txt
        ├── business_entity_sample_100rows.zip
        └── src/
            ├── config.py
            ├── prepare_data.py
            ├── normalization.py
            ├── build_normalized.py
            ├── indic_dictionary.py
            ├── apply_indic_dict.py
            ├── retrieval.py
            ├── candidates.py
            ├── features.py
            ├── build_features.py
            ├── train.py
            ├── predict.py
            ├── metrics.py
            ├── run_all.py
            ├── utils/
            ├── v2/
            └── v4/
```

## 8. Setup

Recommended environment:

``` text
Python 3.12
RAM: 16 GB or more
Disk: 40 GB or more
GPU: Optional
```

``` bash
git clone https://github.com/toru837/Amazon_ml_2026.git
cd Amazon_ml_2026/code/business_entity_resolution
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If CUDA PyTorch is unavailable:

``` bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
grep -vi "^torch" requirements.txt > req_notorch.txt
pip install -r req_notorch.txt
```

On Linux:

``` bash
sudo apt install -y libgomp1
```

## 9. Dataset

The complete challenge dataset is **not included in this repository**
because of file-size limits.

Place it like this:

``` text
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

Set:

``` bash
export ER_DATA_DIR=/path/to/dataset
```

The final result files are generated separately.

## 10. Run the Full Pipeline

``` bash
export ER_DATA_DIR=/path/to/dataset
export ER_CACHE_DIR=$PWD/work/cache
export ER_OUTPUT_DIR=$PWD/work/base_output
export ER_FINAL_OUTPUT_DIR=$PWD/output
export PYTHONPATH=$PWD/src

mkdir -p $ER_CACHE_DIR $ER_OUTPUT_DIR $ER_FINAL_OUTPUT_DIR
```

Step 1:

``` bash
python src/run_all.py
```

Step 2:

``` bash
python src/v2/run_v2.py 0.7
```

Then run the France v4 path from `src/v4/`.

Final files:

``` text
output/
├── candidate_pairs.tsv
└── matching_results.tsv
```

## 11. Quick 100-Row Test

A small sample is included:

``` text
business_entity_sample_100rows.zip
```

Extract:

``` bash
unzip business_entity_sample_100rows.zip -d sample
```

Set:

``` bash
export ER_DATA_DIR=$PWD/sample/business_entity_sample
```

For the tiny sample only, use `min_df=1` and `max_df=1.0` in
`src/retrieval.py`.

The sample only proves that the pipeline runs end to end. Its
perfect-fit metrics are not meaningful for the real competition.

## 12. Resource Usage

  Stage                      Peak Resource
  ---------------------- -----------------
  Normalization                \~5 GiB RAM
  France retrieval         \~1.25 GiB VRAM
  US retrieval             \~2.02 GiB VRAM
  India retrieval          \~2.30 GiB VRAM
  Submission validator         \~10 GB RAM

Tested on a laptop with:

``` text
16 GB RAM
RTX 3050 4 GB
```

For cloud machines, use approximately:

``` text
16 GB+ RAM
40 GB+ disk
GPU optional
```

## 13. Results

The models were evaluated using precision, recall, and the F0.5 score. F0.5 gives more importance to precision.

\[
F_{0.5}=\frac{1.25PR}{0.25P+R}
\]

| Variant | Description | Precision | Recall | F0.5 |
|---|---|---:|---:|---:|
| Earlier Prototype | Sampled corpus | 0.8500 | 0.9850 | **0.8757** |
| v1 | TF-IDF blocking + LightGBM | 0.9407 | 0.9735 | **0.9468** |
| v2 | Indic dictionary + C3/C4 stack (threshold 0.7) | **0.9830** | **0.9890** | **0.9842** |

**Leaderboard:** Rank **1776/27000** — **Team Samadhan**.

**Leaderboard:** rank **1776/27000** for **Team Samadhan**.

## 14. What Worked Well

-   **Country-aware blocking:** fast, high recall, and low memory.
-   **Exact rescoring:** improved candidate quality after approximate
    retrieval.
-   **Indic dictionary:** improved matching for Indian-script records.
-   **Per-country thresholds:** helped control false matches.
-   **C3/C4 stack:** used both pair similarity and candidate-level
    context.
-   **Chunked processing:** reduced memory pressure on large datasets.

## 15. One-Line Summary

**Clean the data → find likely businesses → compare them carefully → let
LightGBM choose → apply thresholds → save final matches.**<img width="672" height="751" alt="image" src="https://github.com/user-attachments/assets/b0e37034-5b46-49be-b6e7-2e7bca0b8b03" /><img width="695" height="856" alt="image" src="https://github.com/user-attachments/assets/ce070cfc-8a11-43cd-9f35-9c57f2225023" />



"""
Learn a transliterated-token -> Latin-token dictionary from the *training* ground truth.

Indic-script names are usually a word-by-word rendering of the Latin Source 1 name
("लक्ष्मी इंजीनियरिंग प्राइवेट लिमिटेड" -> "lakshmi injiniyaring praivet limited" after
rule-based transliteration, vs. "Laxmi Engineering Private Limited"). For matched training
pairs with an Indic-script query name and the same number of tokens on both sides, tokens are
aligned by position; a mapping is kept when it is frequent and dominant. Only training data is
used (no external resources); the dictionary is applied to train and test alike.
"""
import json
import sys
from collections import Counter, defaultdict

import polars as pl

from config import cache_path

DICT_PATH = cache_path("indic_token_dict.json")
MIN_COUNT = 3
MIN_SHARE = 0.6


def learn():
    s1 = pl.read_parquet(cache_path("train_source1_norm.parquet"), columns=["entity_id", "name_norm"])
    def read_q(src):  # rule-based normalization (before any dictionary was applied)
        df = pl.read_parquet(cache_path(f"train_{src}_norm.parquet"))
        col = "name_norm_raw" if "name_norm_raw" in df.columns else "name_norm"
        return df.filter(pl.col("has_indic")).select("entity_id", pl.col(col).alias("name_norm"))
    q = pl.concat([read_q(s) for s in ("source2", "source3")])
    gt = pl.read_parquet(cache_path("train_ground_truth.parquet")).filter(pl.col("matched_entity_ids") != "") \
        .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
    # learn only from S1 entities outside the validation fold (see train.val_fold) so that the
    # validation score is not inflated; test names never contribute.
    gt = gt.filter(~(pl.col("source1_entity_id").str.slice(3).cast(pl.Int64) % 5 == 0))
    pairs = (q.join(gt, left_on="entity_id", right_on="matched_entity_ids")
              .join(s1, left_on="source1_entity_id", right_on="entity_id", suffix="_s1")
              .select("name_norm", "name_norm_s1"))
    pair_counts = Counter()
    tok_counts = Counter()
    for qn, sn in pairs.iter_rows():
        qt, st = qn.split(), sn.split()
        if len(qt) != len(st):
            continue
        for a, b in zip(qt, st):
            tok_counts[a] += 1
            pair_counts[(a, b)] += 1
    best = defaultdict(lambda: (None, 0))
    for (a, b), c in pair_counts.items():
        if c > best[a][1]:
            best[a] = (b, c)
    mapping = {a: b for a, (b, c) in best.items()
               if a != b and c >= MIN_COUNT and c / tok_counts[a] >= MIN_SHARE}
    json.dump(mapping, open(DICT_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(f"aligned pairs: {sum(tok_counts.values()):,} tokens; dictionary entries: {len(mapping):,}")
    return mapping


def load():
    return json.load(open(DICT_PATH, encoding="utf-8"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    m = learn()
    for k in list(m)[:40]:
        print(k, "->", m[k])

"""
Step 5: score the test candidates, assign each S2/S3 record to at most one S1 entity and write
output/matching_results.tsv and output/candidate_pairs.tsv.

Outputs are written in buckets of Source 1 rows so that ~100M candidate pairs never have to be
materialised as strings at once.
"""
import json
import sys
import time

import lightgbm as lgb
import polars as pl

from config import OUTPUT_DIR, cache_path
from train import MODEL_PATH, best_per_query, id_tables, score_split

BUCKET = 200_000  # Source 1 rows per write bucket


def write_grouped(s1_ids: pl.Series, q_ids: pl.Series, pairs: pl.DataFrame, col: str, path):
    """pairs: (s1_idx, q_idx) integer table. Writes one row per S1 entity (file order), comma-joined
    S2/S3 ids sorted, empty when none. Returns (#rows, #non-empty rows)."""
    n_rows = n_nonempty = 0
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(f"source1_entity_id\t{col}\n")
        for s in range(0, len(s1_ids), BUCKET):
            e = min(s + BUCKET, len(s1_ids))
            part = pairs.filter((pl.col("s1_idx") >= s) & (pl.col("s1_idx") < e)).unique()
            part = part.with_columns(q_ids.gather(part["q_idx"]).alias("m_id")).sort("m_id")
            grouped = part.group_by("s1_idx").agg(pl.col("m_id").str.join(","))
            block = (pl.DataFrame({"s1_idx": pl.arange(s, e, eager=True).cast(pl.UInt32),
                                   "source1_entity_id": s1_ids.slice(s, e - s)})
                     .join(grouped.with_columns(pl.col("s1_idx").cast(pl.UInt32)), on="s1_idx", how="left")
                     .sort("s1_idx")
                     .select("source1_entity_id", pl.col("m_id").fill_null("").alias(col)))
            n_rows += block.height
            n_nonempty += block.filter(pl.col(col) != "").height
            fh.write(block.write_csv(separator="\t", quote_style="never", include_header=False))
    return n_rows, n_nonempty


def main(threshold=None):
    t0 = time.time()
    if threshold is None:
        threshold = json.load(open(cache_path("val_best.json")))["threshold"]
    s1, q = id_tables("test")
    s1_ids, q_ids = s1["entity_id"], q["entity_id"]
    del s1, q

    scored_path = cache_path("test_scored.parquet")
    if scored_path.exists():
        scored = pl.read_parquet(scored_path)
    else:
        model = lgb.Booster(model_file=str(MODEL_PATH))
        scored = score_split(model, "test")
        scored.write_parquet(scored_path)
    print(f"scored {scored.height:,} test pairs ({time.time()-t0:.0f}s)", flush=True)

    n, n_c = write_grouped(s1_ids, q_ids, scored.select("s1_idx", "q_idx"), "candidate_entity_ids",
                           OUTPUT_DIR / "candidate_pairs.tsv")
    print(f"candidate_pairs.tsv: {n:,} rows, {n_c:,} with candidates ({time.time()-t0:.0f}s)", flush=True)

    best = best_per_query(scored)
    del scored
    best.write_parquet(cache_path("test_best.parquet"))
    matches = best.filter(pl.col("p1") >= threshold).select("s1_idx", "q_idx")
    n, n_m = write_grouped(s1_ids, q_ids, matches, "matched_entity_ids", OUTPUT_DIR / "matching_results.tsv")
    print(f"matching_results.tsv (threshold={threshold}): {matches.height:,} matched records over "
          f"{n_m:,}/{n:,} S1 entities ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else None)

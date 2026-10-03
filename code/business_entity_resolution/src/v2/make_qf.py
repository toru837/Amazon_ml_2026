"""Build the compact per-record label/fold table once: q_idx, true_s1 (nullable), fold (TRAIN/DEV/CONF)."""
import polars as pl
from runlib import RUNS, Run, cache_path, mem, note
run = Run("v2_data", {"split": "id%5: 0=DEV 1=CONF 2-4=TRAIN", "dict_min_count": 3, "dict_min_share": 0.6})
sp = pl.read_parquet(run.path("split_s1.parquet"))
gt = (pl.read_parquet(cache_path("train_ground_truth.parquet")).filter(pl.col("matched_entity_ids") != "")
      .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids"))
q = pl.concat([pl.read_parquet(cache_path(f"train_{s}.parquet"), columns=["entity_id"]) for s in ("source2", "source3")]).with_row_index("q_idx")
lab = (gt.join(sp.select("s1_idx", "entity_id"), left_on="source1_entity_id", right_on="entity_id")
       .join(q, left_on="matched_entity_ids", right_on="entity_id").select("q_idx", pl.col("s1_idx").alias("true_s1")))
del gt, q
top1 = pl.scan_parquet(str(cache_path("v2train_cands") / "*.parquet")).filter(pl.col("rank") == 1).select("q_idx", pl.col("s1_idx").alias("top1")).collect()
fold = sp.select(pl.col("s1_idx").alias("g"), pl.col("fold").cast(pl.Categorical))
qf = (top1.join(lab, on="q_idx", how="full", coalesce=True).with_columns(pl.coalesce("true_s1", "top1").alias("g"))
      .join(fold, on="g", how="left").select("q_idx", "true_s1", "fold"))
run.write_parquet(qf, "qf.parquet")
run.write_parquet(lab, "truth_pairs.parquet")
note(f"qf.parquet built: {qf.height:,} records; truth pairs {lab.height:,}")
run.release()
print(mem())

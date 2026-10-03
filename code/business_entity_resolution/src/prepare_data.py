"""Step 0: convert the raw TSVs to parquet (fast reloads, exact string preservation)."""
import time
import polars as pl
from config import raw_path, cache_path

SOURCES = ["source1", "source2", "source3"]


def read_tsv(path) -> pl.DataFrame:
    return pl.read_csv(
        path, separator="\t", quote_char=None, has_header=True,
        schema_overrides={c: pl.Utf8 for c in ["entity_id", "business_name", "business_address", "country"]},
    )


def main():
    for split in ["train", "test"]:
        for src in SOURCES:
            t = time.time()
            df = read_tsv(raw_path(split, src)).with_columns(
                pl.col("business_name").fill_null(""), pl.col("business_address").fill_null(""),
                pl.col("country").fill_null(""),
            )
            df.write_parquet(cache_path(f"{split}_{src}.parquet"))
            print(f"{split}_{src}: {df.height:,} rows ({time.time()-t:.1f}s)", flush=True)
    gt = pl.read_csv(raw_path("train", "ground_truth"), separator="\t", quote_char=None,
                     schema_overrides={"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8})
    gt = gt.with_columns(pl.col("matched_entity_ids").fill_null(""))
    gt.write_parquet(cache_path("train_ground_truth.parquet"))
    print(f"ground truth: {gt.height:,} rows")


if __name__ == "__main__":
    main()

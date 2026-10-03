"""Step 3: compute pair features for every candidate chunk of a split -> cache/<split>_feats/*.parquet"""
import sys
import time

import polars as pl

from config import cache_path
from features import compute_features, s1_extra_table

SUB = 1_000_000  # pairs per feature batch (bounded memory)
Q_COLS = ["name_core", "name_norm", "addr_norm", "addr_nums", "is_domain", "has_indic", "addr_missing"]


def load_norm_tables(split: str):
    s1 = pl.read_parquet(cache_path(f"{split}_source1_norm.parquet"),
                         columns=["country", "name_core", "name_norm", "addr_norm", "addr_nums"]).with_row_index("s1_idx")
    q = pl.concat([
        pl.read_parquet(cache_path(f"{split}_source2_norm.parquet"), columns=Q_COLS).with_columns(pl.lit(False).alias("is_s3")),
        pl.read_parquet(cache_path(f"{split}_source3_norm.parquet"), columns=Q_COLS).with_columns(pl.lit(True).alias("is_s3")),
    ]).with_row_index("q_idx")
    return s1, q


def build(split: str):
    cand_dir = cache_path(f"{split}_cands")
    out_dir = cache_path(f"{split}_feats")
    out_dir.mkdir(exist_ok=True)
    s1, q = load_norm_tables(split)
    extra = s1_extra_table(s1, cand_dir)
    files = sorted(cand_dir.glob("*.parquet"))
    t0 = time.time()
    for f in files:
        out = out_dir / f.name
        if out.exists():
            continue
        c = pl.read_parquet(f).sort("q_idx", "rank")
        parts = []
        # split on query boundaries so per-query context features see all candidates of a query
        qids = c["q_idx"].unique(maintain_order=True)
        n_per = max(1, SUB // 10)
        for s in range(0, len(qids), n_per):
            sub = c.filter(pl.col("q_idx").is_in(qids.slice(s, n_per)))
            parts.append(compute_features(sub, q, s1, extra))
        pl.concat(parts).write_parquet(out.with_suffix(".tmp"))
        out.with_suffix(".tmp").replace(out)
        print(f"[{split}] {f.name}: {c.height:,} pairs featurized ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    for sp_ in sys.argv[1:] or ["train", "test"]:
        build(sp_)

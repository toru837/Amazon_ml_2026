"""
Step 1b: rewrite transliterated Indic-script names with the learned token dictionary
(indic_dictionary.py) in the normalized S2/S3 tables of both splits.

Idempotent: the rule-based normalization is kept in `name_norm_raw` and the mapping is
always applied from it.
"""
import sys

import polars as pl

from config import cache_path
from indic_dictionary import load
from normalization import LEGAL_WORDS


def remap(name: str, mapping: dict):
    toks = [mapping.get(t, t) for t in name.split()]
    core = [t for t in toks if t not in LEGAL_WORDS] or toks
    return " ".join(toks), " ".join(core)


def apply(split: str, mapping: dict):
    for src in ("source2", "source3"):
        path = cache_path(f"{split}_{src}_norm.parquet")
        df = pl.read_parquet(path)
        if "name_norm_raw" not in df.columns:
            df = df.with_columns(pl.col("name_norm").alias("name_norm_raw"))
        mask = df["has_indic"].to_numpy()
        raw = df["name_norm_raw"].to_list()
        norm, core = df["name_norm"].to_list(), df["name_core"].to_list()
        changed = 0
        for i in mask.nonzero()[0]:
            n, c = remap(raw[i], mapping)
            changed += n != norm[i]
            norm[i], core[i] = n, c
        df = df.with_columns(pl.Series("name_norm", norm), pl.Series("name_core", core))
        df.write_parquet(path)
        print(f"{split}_{src}: {int(mask.sum()):,} Indic-script names, {changed:,} changed by dictionary")


if __name__ == "__main__":
    m = load()
    for sp_ in sys.argv[1:] or ["train", "test"]:
        apply(sp_, m)

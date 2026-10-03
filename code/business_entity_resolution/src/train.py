"""
Step 4: train the pair-matching model on the training split and validate it end-to-end.

Validation protocol (mirrors the test setting):
  * Source 1 entities are split into folds A (80%, train) and B (20%, validation) by entity id.
  * Every S2/S3 query record is assigned to the fold of its true S1 entity, or - when it has no
    match - to the fold of its rank-1 retrieved S1. The model is fit only on fold-A queries.
  * The model then scores *all* candidate pairs, each query is assigned to its best S1 when the
    probability clears the threshold, and macro F0.5 is computed over fold-B S1 entities using
    every assignment (exactly the leaderboard metric).
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from config import SEED, cache_path
from features import FEATURES
from metrics import macro_f05

TRAIN_QUERY_FRACTION = 0.10  # fraction of fold-A queries used to fit the model (~8M pairs)
MODEL_PATH = cache_path("lgbm_model.txt")

PARAMS = dict(
    objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=200,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    max_bin=255, num_threads=16, seed=SEED, verbose=-1,
)
N_ROUNDS = 1500


def id_tables(split: str):
    s1 = pl.read_parquet(cache_path(f"{split}_source1.parquet"), columns=["entity_id", "country"]).with_row_index("s1_idx")
    q = pl.concat([pl.read_parquet(cache_path(f"{split}_source2.parquet"), columns=["entity_id"]),
                   pl.read_parquet(cache_path(f"{split}_source3.parquet"), columns=["entity_id"])]).with_row_index("q_idx")
    return s1, q


def truth_pairs() -> pl.DataFrame:
    gt = pl.read_parquet(cache_path("train_ground_truth.parquet"))
    return (gt.filter(pl.col("matched_entity_ids") != "")
            .with_columns(pl.col("matched_entity_ids").str.split(","))
            .explode("matched_entity_ids")
            .select(pl.col("source1_entity_id").alias("s1_id"), pl.col("matched_entity_ids").alias("m_id")))


def val_fold(ids: pl.Series) -> pl.Series:
    """Deterministic 20% fold from the numeric part of the entity id."""
    return (ids.str.slice(3).cast(pl.Int64) % 5 == 0)


def label_table():
    """q_idx -> true s1_idx (null if the query has no match), and s1 fold flags."""
    s1, q = id_tables("train")
    tp = truth_pairs()
    lab = (tp.join(s1.select("entity_id", "s1_idx"), left_on="s1_id", right_on="entity_id")
             .join(q.select("entity_id", "q_idx"), left_on="m_id", right_on="entity_id")
             .select("q_idx", pl.col("s1_idx").alias("true_s1")))
    s1 = s1.with_columns(val_fold(s1["entity_id"]).alias("is_val"))
    return s1, q, lab, tp


def query_folds(lab: pl.DataFrame, s1: pl.DataFrame) -> pl.DataFrame:
    top1 = (pl.scan_parquet(str(cache_path("train_cands") / "*.parquet")).filter(pl.col("rank") == 1)
            .select("q_idx", pl.col("s1_idx").alias("top1")).collect())
    qf = top1.join(lab, on="q_idx", how="full", coalesce=True).with_columns(
        pl.coalesce("true_s1", "top1").alias("group_s1"))
    return qf.join(s1.select(pl.col("s1_idx").alias("group_s1"), pl.col("is_val").alias("q_is_val")),
                   on="group_s1", how="left").select("q_idx", "true_s1", "q_is_val")


def load_training_rows(qf: pl.DataFrame):
    """Returns (X_train, y_train, X_es, y_es) as numpy. Early-stopping rows are other fold-A queries,
    so fold B stays untouched until the final evaluation."""
    keep_tr = (pl.col("q_idx").hash(SEED) % 1000) < int(TRAIN_QUERY_FRACTION * 1000)
    keep_es = ~keep_tr & ((pl.col("q_idx").hash(SEED + 1) % 1000) < 20)
    Xtr, ytr, Xes, yes = [], [], [], []
    for f in sorted(cache_path("train_feats").glob("*.parquet")):
        d = pl.read_parquet(f).join(qf, on="q_idx", how="left").with_columns(
            (pl.col("s1_idx") == pl.col("true_s1")).fill_null(False).alias("y"))
        a = d.filter(~pl.col("q_is_val") & keep_tr)
        b = d.filter(~pl.col("q_is_val") & keep_es)
        del d
        Xtr.append(a.select(FEATURES).to_numpy().astype(np.float32)); ytr.append(a["y"].to_numpy())
        Xes.append(b.select(FEATURES).to_numpy().astype(np.float32)); yes.append(b["y"].to_numpy())
    return np.concatenate(Xtr), np.concatenate(ytr), np.concatenate(Xes), np.concatenate(yes)


def fit(Xtr, ytr, Xes, yes):
    dtr = lgb.Dataset(Xtr, label=ytr, feature_name=FEATURES, free_raw_data=True)
    dva = lgb.Dataset(Xes, label=yes, reference=dtr)
    model = lgb.train(PARAMS, dtr, N_ROUNDS, valid_sets=[dva], valid_names=["val"],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    return model


def score_split(model, split: str) -> pl.DataFrame:
    """Scores every candidate pair; returns all pairs (q_idx, s1_idx, p) sorted by q and p."""
    outs = []
    for f in sorted(cache_path(f"{split}_feats").glob("*.parquet")):
        d = pl.read_parquet(f)
        p = model.predict(d.select(FEATURES).to_numpy(), num_threads=16)
        outs.append(d.select("q_idx", "s1_idx").with_columns(pl.Series("p", p.astype(np.float32))))
    return pl.concat(outs)


def best_per_query(scored: pl.DataFrame) -> pl.DataFrame:
    return (scored.sort(["q_idx", "p"], descending=[False, True])
            .group_by("q_idx", maintain_order=True)
            .agg(pl.col("s1_idx").first(), pl.col("p").first().alias("p1"),
                 pl.col("p").slice(1, 1).first().fill_null(0.0).alias("p2")))


def evaluate(best: pl.DataFrame, s1: pl.DataFrame, q: pl.DataFrame, tp: pl.DataFrame, thresholds):
    val_s1 = s1.filter(pl.col("is_val"))
    ids = best.join(s1.select("s1_idx", pl.col("entity_id").alias("s1_id"), "country"), on="s1_idx") \
              .join(q.select("q_idx", pl.col("entity_id").alias("m_id")), on="q_idx")
    res = []
    for t in thresholds:
        pred = ids.filter(pl.col("p1") >= t).select("s1_id", "m_id")
        m = macro_f05(pred, tp, val_s1["entity_id"])
        row = {"threshold": t, "macro_f05": m["macro_f05"], "precision": m["macro_precision"],
               "recall": m["macro_recall"], "singleton_acc": m["singleton_acc"]}
        for c in val_s1["country"].unique().sort().to_list():
            mc = macro_f05(pred, tp, val_s1.filter(pl.col("country") == c)["entity_id"])
            row[f"f05_{c}"] = mc["macro_f05"]
        res.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    return pl.DataFrame(res)


def main():
    t0 = time.time()
    s1, q, lab, tp = label_table()
    qf = query_folds(lab, s1)
    print(f"labels ready ({time.time()-t0:.0f}s): {lab.height:,} true links, "
          f"{qf.filter(pl.col('q_is_val')).height:,} val-fold queries", flush=True)
    Xtr, ytr, Xes, yes = load_training_rows(qf)
    del qf
    print(f"train rows {len(ytr):,} (pos {ytr.sum():,}); early-stop rows {len(yes):,} ({time.time()-t0:.0f}s)", flush=True)
    model = fit(Xtr, ytr, Xes, yes)
    del Xtr, ytr, Xes, yes
    model.save_model(str(MODEL_PATH))
    imp = sorted(zip(FEATURES, model.feature_importance("gain")), key=lambda x: -x[1])
    print("top features:", [(f, round(g)) for f, g in imp[:15]])
    scored = score_split(model, "train")
    scored.write_parquet(cache_path("train_scored.parquet"))
    best = best_per_query(scored)
    del scored
    best.write_parquet(cache_path("train_best.parquet"))
    print(f"scored all train pairs ({time.time()-t0:.0f}s)", flush=True)
    res = evaluate(best, s1, q, tp, [0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    res.write_csv(cache_path("val_threshold_sweep.csv"))
    bestrow = res.sort("macro_f05", descending=True).row(0, named=True)
    json.dump(bestrow, open(cache_path("val_best.json"), "w"), indent=2)
    print("BEST", bestrow)


if __name__ == "__main__":
    main()

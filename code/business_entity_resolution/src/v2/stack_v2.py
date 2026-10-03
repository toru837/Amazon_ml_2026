"""C4: per-record second stage on top of a first-stage direct model (no sibling / population inputs).
Row = each record's best candidate. Features = first-stage features of that pair + the record's own top-3
first-stage probabilities and margin (competition among S1 candidates only -> duplication-invariant).
Trained on TRAIN-fold records that the first stage did NOT train or early-stop on (out-of-sample inputs).
usage: python stack_v2.py <first_stage_config> [split_to_score ...]   (fit happens once on v2train)"""
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from runlib import RUNS, Run, cache_path, mem, note
import configs_v2 as T1

first = sys.argv[1]
splits = sys.argv[2:] or ["v2train"]
cfg1 = T1.CONFIGS[first]
FEATS1 = cfg1["feats"]
S2F = FEATS1 + ["p1", "p2", "p3", "margin12"]
SEED = 42
run = Run(f"model_C4_stack_on_{first}", {"first": first, "feats": S2F, "leaves": 127}, code_files=[__file__])


def best_rows(split):
    """yield (features of each record's best pair + p1,p2,p3,margin) in bounded batches"""
    top = pl.read_parquet(RUNS / f"model_{first}" / f"top3_{split}.parquet").select(
        "q_idx", "s1_idx", "p1", "p2", "p3", (pl.col("p1") - pl.col("p2")).alias("margin12"))
    fdir, xdir = cache_path(f"{split}_feats"), cache_path(f"{split}_featx")
    base_cols = ["q_idx", "s1_idx"] + [c for c in FEATS1 if c not in T1.XCOLS]
    for f in sorted(fdir.glob("*.parquet")):
        n, lo, hi = pl.scan_parquet(f).select(pl.len().alias("n"), pl.col("q_idx").min().alias("lo"), pl.col("q_idx").max().alias("hi")).collect().row(0)
        k = max(1, -(-n // 1_500_000))
        edges = [lo + (hi + 1 - lo) * i // k for i in range(k + 1)]
        for a, b in zip(edges[:-1], edges[1:]):
            rng = (pl.col("q_idx") >= a) & (pl.col("q_idx") < b)
            t = top.filter(rng)
            d = pl.scan_parquet(f).select(base_cols).filter(rng).collect().join(t, on=["q_idx", "s1_idx"], how="inner")
            if cfg1["extra"]:
                d = d.join(pl.scan_parquet(xdir / f.name).filter(rng).collect(), on=["q_idx", "s1_idx"], how="left")
            yield d


if not run.done("fit"):
    qf = pl.read_parquet(RUNS / "v2_data" / "qf.parquet")
    h1 = pl.col("q_idx").hash(SEED) % 1000
    used_by_first = (h1 < int(cfg1["frac"] * 1000)) | ((pl.col("q_idx").hash(SEED + 1) % 1000) < 15)
    pool = (pl.col("fold") == "TRAIN") & ~used_by_first
    es = (pl.col("q_idx").hash(SEED + 5) % 100) < 4
    Xa, ya, Xe, ye = [], [], [], []
    for d in best_rows("v2train"):
        d = d.join(qf, on="q_idx", how="left").with_columns((pl.col("s1_idx") == pl.col("true_s1")).fill_null(False).alias("y"))
        a = d.filter(pool & ~es).sample(fraction=0.6, seed=SEED)
        b = d.filter(pool & es)
        Xa.append(a.select(S2F).to_numpy().astype(np.float32)); ya.append(a["y"].to_numpy())
        Xe.append(b.select(S2F).to_numpy().astype(np.float32)); ye.append(b["y"].to_numpy())
    Xa, ya, Xe, ye = map(np.concatenate, (Xa, ya, Xe, ye))
    print(f"stack train rows {len(ya):,} (pos {ya.mean():.3f}) es {len(ye):,}", mem(), flush=True)
    dtr = lgb.Dataset(Xa, ya, feature_name=S2F, free_raw_data=True).construct()
    dva = lgb.Dataset(Xe, ye, reference=dtr).construct()
    del Xa, ya, Xe, ye
    params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=16, seed=SEED, verbose=-1)
    t0 = time.time()
    m = lgb.train(params, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)])
    m.save_model(str(run.path("model.txt")))
    imp = sorted(zip(S2F, m.feature_importance("gain")), key=lambda t: -t[1])[:12]
    run.complete("fit", files=["model.txt"], best_iter=m.best_iteration, fit_sec=round(time.time() - t0),
                 top_features=[(k, round(float(v))) for k, v in imp])
    print("stack fit done", m.best_iteration, imp[:8], flush=True)

m = lgb.Booster(model_file=str(run.path("model.txt")))
for split in splits:
    if run.done(f"score_{split}"):
        continue
    parts = []
    for d in best_rows(split):
        parts.append(d.select("q_idx", "s1_idx", "p2", "p3").with_columns(
            pl.Series("p1", m.predict(d.select(S2F).to_numpy(), num_threads=16).astype(np.float32))))
    top = pl.concat(parts).select("q_idx", "s1_idx", "p1", "p2", "p3")
    run.write_parquet(top, f"top3_{split}.parquet")
    run.complete(f"score_{split}", files=[f"top3_{split}.parquet"])
    note(f"C4 stack on {first}: scored {split} ({top.height:,} records)")
run.release()

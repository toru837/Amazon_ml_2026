"""v2 pairwise model: fit on TRAIN-fold records only, score every candidate pair, keep top-3 per record.
usage: python train_v2.py <config_name>
Every config uses only direct record<->S1 evidence (plus the baseline's own features for B0)."""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from runlib import Run, cache_path, mem, note
from features import FEATURES

from configs_v2 import CONFIGS, NO_POP, XCOLS  # noqa: E402
SEED = 42
name = sys.argv[1]
cfg = CONFIGS[name]
split = sys.argv[2] if len(sys.argv) > 2 else "v2train"   # model is always FIT on v2train; other splits are scored only
run = Run(f"model_{name}", cfg, code_files=[__file__])
FEATS = cfg["feats"]
def dirs(sp):
    return cache_path(f"{sp}_feats"), cache_path(f"{sp}_featx")


feat_dir, x_dir = dirs("v2train")
files = sorted(feat_dir.glob("*.parquet"))


def frame_batches(f, x_dir=x_dir, max_rows=1_500_000):
    """yield feature frames for a candidate file in q_idx-range batches (bounded memory)"""
    base_cols = ["q_idx", "s1_idx"] + [c for c in FEATS if c not in XCOLS]
    info = pl.scan_parquet(f).select(pl.len().alias("n"), pl.col("q_idx").min().alias("lo"), pl.col("q_idx").max().alias("hi")).collect().row(0)
    n, lo, hi = info
    k = max(1, -(-n // max_rows))
    edges = [lo + (hi + 1 - lo) * i // k for i in range(k + 1)]
    for a, b in zip(edges[:-1], edges[1:]):
        rng = (pl.col("q_idx") >= a) & (pl.col("q_idx") < b)
        d = pl.scan_parquet(f).select(base_cols).filter(rng).collect()
        if cfg["extra"]:
            d = d.join(pl.scan_parquet(x_dir / f.name).filter(rng).collect(), on=["q_idx", "s1_idx"], how="left")
        yield d


if not run.done("fit"):
    qf = pl.read_parquet(cache_path("runs") / "v2_data" / "qf.parquet")  # q_idx, true_s1, fold (built once by make_qf.py)
    keep_tr = (pl.col("fold") == "TRAIN") & ((pl.col("q_idx").hash(SEED) % 1000) < int(cfg["frac"] * 1000))
    keep_es = (pl.col("fold") == "TRAIN") & ~keep_tr & ((pl.col("q_idx").hash(SEED + 1) % 1000) < 15)
    # count, then fill preallocated arrays
    n_tr = n_es = 0
    for f in files:
        d = pl.read_parquet(f, columns=["q_idx"]).join(qf, on="q_idx", how="left")
        n_tr += d.filter(keep_tr).height; n_es += d.filter(keep_es).height
    Xa = np.empty((n_tr, len(FEATS)), np.float32); ya = np.empty(n_tr, np.float32)
    Xe = np.empty((n_es, len(FEATS)), np.float32); ye = np.empty(n_es, np.float32)
    ia = ie = 0
    for f in files:
        for d in frame_batches(f):
            d = d.join(qf, on="q_idx", how="left").with_columns((pl.col("s1_idx") == pl.col("true_s1")).fill_null(False).alias("y"))
            a, b = d.filter(keep_tr), d.filter(keep_es)
            Xa[ia:ia + a.height] = a.select(FEATS).to_numpy(); ya[ia:ia + a.height] = a["y"].to_numpy(); ia += a.height
            Xe[ie:ie + b.height] = b.select(FEATS).to_numpy(); ye[ie:ie + b.height] = b["y"].to_numpy(); ie += b.height
    print(f"train rows {n_tr:,} (pos {ya.mean():.3f}) es rows {n_es:,}", mem(), flush=True)
    params = dict(objective="binary", learning_rate=0.05, num_leaves=cfg["leaves"], min_data_in_leaf=200, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255, num_threads=16, seed=SEED, verbose=-1)
    t0 = time.time()
    dtr = lgb.Dataset(Xa, ya, feature_name=FEATS, free_raw_data=True, params={"max_bin": 255}).construct()
    dva = lgb.Dataset(Xe, ye, reference=dtr).construct()
    del Xa, ya, Xe, ye  # binned copies live inside LightGBM; release the float arrays before training
    print("datasets constructed", mem(), flush=True)
    m = lgb.train(params, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(250)])
    m.save_model(str(run.path("model.txt")))
    imp = sorted(zip(FEATS, m.feature_importance("gain")), key=lambda t: -t[1])[:15]
    run.complete("fit", files=["model.txt"], best_iter=m.best_iteration, fit_sec=round(time.time() - t0),
                 top_features=[(k, round(float(v))) for k, v in imp])
    print("fit done", m.best_iteration, imp[:8], flush=True)

if not run.done(f"score_{split}"):
    m = lgb.Booster(model_file=str(run.path("model.txt")))
    parts = []
    sf, sx = dirs(split)
    for f in sorted(sf.glob("*.parquet")):
        for d in frame_batches(f, sx):
            p = m.predict(d.select(FEATS).to_numpy(), num_threads=16).astype(np.float32)
            s = d.select("q_idx", "s1_idx").with_columns(pl.Series("p", p))
            parts.append(s.sort(["q_idx", "p"], descending=[False, True]).group_by("q_idx", maintain_order=True).agg(
                pl.col("s1_idx").head(3).alias("s"), pl.col("p").head(3).alias("pp")))
    top = pl.concat(parts).with_columns(
        pl.col("s").list.get(0).alias("s1_idx"), pl.col("pp").list.get(0).alias("p1"),
        pl.col("s").list.get(1, null_on_oob=True).alias("s1_2"), pl.col("pp").list.get(1, null_on_oob=True).fill_null(0.0).alias("p2"),
        pl.col("pp").list.get(2, null_on_oob=True).fill_null(0.0).alias("p3")).drop("s", "pp")
    run.write_parquet(top, f"top3_{split}.parquet")
    run.complete(f"score_{split}", files=[f"top3_{split}.parquet"])
note(f"{name} on {split}: fit+score complete ({run.m['stages']['fit'].get('best_iter')} trees)")
run.release()

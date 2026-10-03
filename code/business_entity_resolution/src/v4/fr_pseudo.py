"""Self-training for France (validated first on the US->India proxy, see tx_pseudo.py).
Labelled: US+India TRAIN-fold rows (C3 features, 15% of TRAIN records). Pseudo: France TEST records from the
France-locale split (v4test) whose C3 top-1 probability is >= HI (top-1 = match, other candidates = non-match)
or <= LO (all candidates non-match). No labels exist for France; none are used.
The fitted model scores only France records; US/India decisions stay with C4.
usage: python fr_pseudo.py <name> [HI] [LO] [PFRAC] [LFRAC]"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "v2"))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from configs_v2 import CONFIGS, XCOLS  # noqa: E402
from runlib import RUNS, Run, cache_path, mem, note  # noqa: E402

name = sys.argv[1]
HI = float(sys.argv[2]) if len(sys.argv) > 2 else 0.98
LO = float(sys.argv[3]) if len(sys.argv) > 3 else 0.02
PFRAC = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25
LFRAC = float(sys.argv[5]) if len(sys.argv) > 5 else 0.15
FEATS = CONFIGS["C3_direct_x_more_data"]["feats"]
SEED = 42
run = Run(f"frp_{name}", {"hi": HI, "lo": LO, "pfrac": PFRAC, "lfrac": LFRAC}, code_files=[__file__])


def batches(split, prefix):
    base_cols = ["q_idx", "s1_idx"] + [c for c in FEATS if c not in XCOLS]
    for f in sorted(cache_path(f"{split}_feats").glob(f"{prefix}*.parquet")):
        n, lo, hi = pl.scan_parquet(f).select(pl.len().alias("n"), pl.col("q_idx").min().alias("lo"), pl.col("q_idx").max().alias("hi")).collect().row(0)
        k = max(1, -(-n // 1_500_000))
        edges = [lo + (hi + 1 - lo) * i // k for i in range(k + 1)]
        for a, b in zip(edges[:-1], edges[1:]):
            rng = (pl.col("q_idx") >= a) & (pl.col("q_idx") < b)
            d = pl.scan_parquet(f).select(base_cols).filter(rng).collect()
            yield d.join(pl.scan_parquet(cache_path(f"{split}_featx") / f.name).filter(rng).collect(), on=["q_idx", "s1_idx"], how="left")


if not run.done("fit"):
    qf = pl.read_parquet(RUNS / "v2_data" / "qf.parquet")
    keep = (pl.col("fold") == "TRAIN") & ((pl.col("q_idx").hash(SEED) % 1000) < int(LFRAC * 1000))
    es = (pl.col("fold") == "TRAIN") & ~keep & ((pl.col("q_idx").hash(SEED + 1) % 1000) < 15)
    top = (pl.read_parquet(RUNS / "model_C3_direct_x_more_data" / "top3_v4test.parquet").select("q_idx", pl.col("s1_idx").alias("top_s1"), pl.col("p1").alias("top_p")))
    pick = (pl.col("q_idx").hash(SEED + 3) % 1000) < int(PFRAC * 1000)

    def lab_frames():
        for d in batches("v2train", ""):
            d = d.join(qf, on="q_idx", how="left").with_columns((pl.col("s1_idx") == pl.col("true_s1")).fill_null(False).alias("y"))
            yield d.filter(keep), d.filter(es)

    def fr_frames():
        for d in batches("v4test", "France"):
            d = d.filter(pick).join(top, on="q_idx", how="inner")
            yield d.filter((pl.col("top_p") >= HI) | (pl.col("top_p") <= LO)).with_columns(
                ((pl.col("top_p") >= HI) & (pl.col("s1_idx") == pl.col("top_s1"))).alias("y"))

    n_tr = n_es = n_ps = 0
    for a_, b_ in lab_frames():
        n_tr += a_.height; n_es += b_.height
    for d in fr_frames():
        n_ps += d.height
    print(f"rows: labelled {n_tr:,}, es {n_es:,}, France pseudo {n_ps:,}", mem(), flush=True)
    Xa = np.empty((n_tr + n_ps, len(FEATS)), np.float32); ya = np.empty(n_tr + n_ps, np.float32)
    Xe = np.empty((n_es, len(FEATS)), np.float32); ye = np.empty(n_es, np.float32)
    ia = ie = 0
    for a_, b_ in lab_frames():
        Xa[ia:ia + a_.height] = a_.select(FEATS).to_numpy(); ya[ia:ia + a_.height] = a_["y"].to_numpy(); ia += a_.height
        Xe[ie:ie + b_.height] = b_.select(FEATS).to_numpy(); ye[ie:ie + b_.height] = b_["y"].to_numpy(); ie += b_.height
    npos = 0
    for d in fr_frames():
        Xa[ia:ia + d.height] = d.select(FEATS).to_numpy(); ya[ia:ia + d.height] = d["y"].to_numpy(); ia += d.height
        npos += int(d["y"].sum())
    assert ia == n_tr + n_ps and ie == n_es
    params = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=200, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255, num_threads=12, seed=SEED, verbose=-1)
    dtr = lgb.Dataset(Xa, ya, feature_name=FEATS, free_raw_data=True).construct()
    dva = lgb.Dataset(Xe, ye, reference=dtr).construct()   # early stopping on labelled rows only
    del Xa, ya, Xe, ye
    print("datasets constructed", mem(), flush=True)
    m = lgb.train(params, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(250)])
    m.save_model(str(run.path("model.txt")))
    run.complete("fit", files=["model.txt"], trees=m.best_iteration, labelled=n_tr, pseudo=n_ps, pseudo_pos=npos)
    note(f"FR-PSEUDO {name}: fit {m.best_iteration} trees; labelled {n_tr:,}, France pseudo rows {n_ps:,} ({npos:,} positive)")

if not run.done("score"):
    m = lgb.Booster(model_file=str(run.path("model.txt")))
    parts = []
    for d in batches("v4test", "France"):
        p = m.predict(d.select(FEATS).to_numpy(), num_threads=12).astype(np.float32)
        s = d.select("q_idx", "s1_idx").with_columns(pl.Series("p", p))
        parts.append(s.sort(["q_idx", "p"], descending=[False, True]).group_by("q_idx", maintain_order=True).agg(
            pl.col("s1_idx").first(), pl.col("p").first().alias("p1")))
    t = pl.concat(parts)
    run.write_parquet(t, "top1_france.parquet")
    run.complete("score", files=["top1_france.parquet"])
    note(f"FR-PSEUDO {name}: scored {t.height:,} France records; accept@.7 {(t['p1'] >= .7).mean():.4f} @.8 {(t['p1'] >= .8).mean():.4f}")
run.release()

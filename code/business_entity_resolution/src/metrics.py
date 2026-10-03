"""Macro F0.5 exactly as defined by the challenge (per Source 1 entity, singletons included)."""
import polars as pl


def macro_f05(pred: pl.DataFrame, truth: pl.DataFrame, eval_ids: pl.Series, beta: float = 0.5) -> dict:
    """pred / truth: long pairs tables (s1_id, m_id). eval_ids: S1 entities to average over."""
    b2 = beta * beta
    ev = pl.DataFrame({"s1_id": eval_ids})
    pred = pred.join(ev, on="s1_id", how="semi").unique()
    truth = truth.join(ev, on="s1_id", how="semi").unique()
    tp = pred.join(truth, on=["s1_id", "m_id"], how="inner").group_by("s1_id").agg(pl.len().alias("tp"))
    np_ = pred.group_by("s1_id").agg(pl.len().alias("n_pred"))
    nt = truth.group_by("s1_id").agg(pl.len().alias("n_true"))
    df = ev.join(tp, on="s1_id", how="left").join(np_, on="s1_id", how="left").join(nt, on="s1_id", how="left")
    df = df.fill_null(0).with_columns(
        pl.when(pl.col("n_pred") > 0).then(pl.col("tp") / pl.col("n_pred"))
          .otherwise(pl.when(pl.col("n_true") == 0).then(1.0).otherwise(0.0)).alias("p"),
        pl.when(pl.col("n_true") > 0).then(pl.col("tp") / pl.col("n_true"))
          .otherwise(pl.when(pl.col("n_pred") == 0).then(1.0).otherwise(0.0)).alias("r"),
    ).with_columns(
        pl.when((pl.col("n_true") == 0) & (pl.col("n_pred") == 0)).then(1.0)
          .when((pl.col("p") + pl.col("r")) == 0).then(0.0)
          .otherwise((1 + b2) * pl.col("p") * pl.col("r") / (b2 * pl.col("p") + pl.col("r"))).alias("f")
    )
    single = df.filter(pl.col("n_true") == 0)
    return {
        "macro_f05": df["f"].mean(),
        "macro_precision": df["p"].mean(),
        "macro_recall": df["r"].mean(),
        "n_entities": df.height,
        "singleton_acc": single["f"].mean() if single.height else None,
        "per_entity": df,
    }

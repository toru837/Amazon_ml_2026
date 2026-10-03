"""
Step 3: pairwise features for (query S2/S3 record, candidate S1 record).

All features are country-agnostic similarity / context signals (no country one-hot), so the
model transfers to countries unseen in training (France in test).
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

WORKERS = -1

FEATURES = [
    # retrieval scores / query context
    "cos_name", "cos_addr", "cos_comb", "rank", "n_cands", "comb_max", "comb_gap", "comb_gap_next",
    "name_gap", "addr_gap",
    # name similarity
    "nm_ratio", "nm_tset", "nm_tsort", "nm_partial", "key_ratio", "key_partial", "key_jw", "full_ratio",
    "nm_exact", "key_exact", "tok_inter", "tok_jacc", "tok_q_cov", "tok_s_cov", "first_tok_eq",
    "len_q", "len_s", "len_ratio",
    # address similarity
    "ad_ratio", "ad_tset", "ad_tsort", "ad_partial", "ad_jacc", "ad_q_cov",
    "num_q", "num_s", "num_inter", "num_q_cov", "num_first_eq", "num_conflict",
    "q_addr_missing", "ad_len_q", "ad_len_s",
    # record flags / ambiguity
    "q_is_s3", "q_domain", "q_indic", "s1_name_dup", "s1_rank1_deg",
    # what differs: tokens / numbers present on one side only (hard negatives differ by one word or number)
    "nm_q_only", "nm_s_only", "nm_diff_ratio", "ad_q_only", "ad_qonly_in_s", "num_q_only", "num_s_only",
    "num_rel_diff", "num_set_eq",
    # margins versus the best *other* candidate of the same query
    "m_cos_name", "m_cos_addr", "m_nm_tset", "m_full_ratio", "m_ad_tset", "m_num_q_cov", "n_name_ties",
]
MARGIN_COLS = {"m_cos_name": "cos_name", "m_cos_addr": "cos_addr", "m_nm_tset": "nm_tset",
               "m_full_ratio": "full_ratio", "m_ad_tset": "ad_tset", "m_num_q_cov": "num_q_cov"}


def _cp(scorer, a, b, **kw):
    return process.cpdist(a, b, scorer=scorer, workers=WORKERS, dtype=np.float32, **kw)


def add_query_context(c: pl.DataFrame) -> pl.DataFrame:
    """Context of each candidate relative to the other candidates of the same query."""
    return c.with_columns(
        pl.len().over("q_idx").cast(pl.Float32).alias("n_cands"),
        pl.col("cos_comb").max().over("q_idx").alias("comb_max"),
    ).with_columns(
        # gap to the best *other* candidate (negative if another S1 is better)
        (pl.col("cos_comb") - pl.when(pl.col("rank") == 1)
         .then(pl.col("cos_comb").filter(pl.col("rank") == 2).first().over("q_idx"))
         .otherwise(pl.col("comb_max"))).fill_null(pl.col("cos_comb")).alias("comb_gap"),
        (pl.col("cos_comb") - pl.col("cos_comb").sort(descending=True).shift(-1).over("q_idx", order_by="rank")
         ).fill_null(pl.col("cos_comb")).alias("comb_gap_next"),
        (pl.col("cos_name") - pl.col("cos_name").max().over("q_idx")).alias("name_gap"),
        (pl.col("cos_addr") - pl.col("cos_addr").max().over("q_idx")).alias("addr_gap"),
    )


def compute_features(c: pl.DataFrame, q: pl.DataFrame, s1: pl.DataFrame, s1_extra: pl.DataFrame) -> pl.DataFrame:
    """c: candidate pairs (q_idx, s1_idx, cos_*, rank). q / s1: normalized tables indexed by q_idx / s1_idx.
    s1_extra: (s1_idx, s1_name_dup, s1_rank1_deg)."""
    c = add_query_context(c)
    qi = c["q_idx"].to_numpy()
    si = c["s1_idx"].to_numpy()
    Q = q.select("name_core", "name_norm", "addr_norm", "addr_nums", "is_domain", "has_indic",
                 "addr_missing", "is_s3")[qi]
    S = s1.select("name_core", "name_norm", "addr_norm", "addr_nums")[si]
    qc, sc = Q["name_core"].to_list(), S["name_core"].to_list()
    qk = Q["name_core"].str.replace_all(" ", "").to_list()
    sk = S["name_core"].str.replace_all(" ", "").to_list()
    qa, sa = Q["addr_norm"].to_list(), S["addr_norm"].to_list()

    f = {
        "nm_ratio": _cp(fuzz.ratio, qc, sc),
        "nm_tset": _cp(fuzz.token_set_ratio, qc, sc),
        "nm_tsort": _cp(fuzz.token_sort_ratio, qc, sc),
        "nm_partial": _cp(fuzz.partial_ratio, qc, sc),
        "key_ratio": _cp(fuzz.ratio, qk, sk),
        "key_partial": _cp(fuzz.partial_ratio, qk, sk),
        "key_jw": _cp(JaroWinkler.normalized_similarity, qk, sk),
        "full_ratio": _cp(fuzz.ratio, Q["name_norm"].to_list(), S["name_norm"].to_list()),
        "ad_ratio": _cp(fuzz.ratio, qa, sa),
        "ad_tset": _cp(fuzz.token_set_ratio, qa, sa),
        "ad_tsort": _cp(fuzz.token_sort_ratio, qa, sa),
        "ad_partial": _cp(fuzz.partial_ratio, qa, sa),
    }
    del qc, sc, qk, sk, qa, sa

    tok = pl.DataFrame({
        "qt": Q["name_core"].str.split(" "), "st": S["name_core"].str.split(" "),
        "qa": Q["addr_norm"].str.split(" "), "sa": S["addr_norm"].str.split(" "),
        "qn": Q["addr_nums"].str.extract_all(r"\d+"), "sn": S["addr_nums"].str.extract_all(r"\d+"),
    }).select(
        pl.col("qt").list.set_intersection("st").list.len().alias("tok_inter"),
        pl.col("qt").list.set_union("st").list.len().alias("tok_union"),
        pl.col("qt").list.unique().list.len().alias("q_ntok"),
        pl.col("st").list.unique().list.len().alias("s_ntok"),
        (pl.col("qt").list.first() == pl.col("st").list.first()).alias("first_tok_eq"),
        pl.col("qa").list.set_intersection("sa").list.len().alias("ad_inter"),
        pl.col("qa").list.set_union("sa").list.len().alias("ad_union"),
        pl.col("qa").list.unique().list.len().alias("qa_n"),
        pl.col("qn").list.unique().list.len().alias("num_q"),
        pl.col("sn").list.unique().list.len().alias("num_s"),
        pl.col("qn").list.set_intersection("sn").list.len().alias("num_inter"),
        (pl.col("qn").list.first() == pl.col("sn").list.first()).fill_null(False).alias("num_first_eq"),
        pl.col("qt").list.set_difference("st").list.sort().list.join(" ").alias("nm_q_diff"),
        pl.col("st").list.set_difference("qt").list.sort().list.join(" ").alias("nm_s_diff"),
        pl.col("qa").list.set_difference("sa").list.join(" ").alias("ad_q_diff"),
        pl.col("qn").list.set_difference("sn").list.len().alias("num_q_only"),
        pl.col("sn").list.set_difference("qn").list.len().alias("num_s_only"),
        pl.col("qn").list.first().cast(pl.Float64, strict=False).alias("qn1"),
        pl.col("sn").list.first().cast(pl.Float64, strict=False).alias("sn1"),
        (pl.col("qn").list.unique().list.sort() == pl.col("sn").list.unique().list.sort()).alias("num_set_eq"),
    )
    # similarity of the leftover (non-shared) name tokens: typo (high) vs substituted word (low)
    qd, sd = tok["nm_q_diff"].to_list(), tok["nm_s_diff"].to_list()
    nm_diff_ratio = _cp(fuzz.ratio, qd, sd)
    both_empty = (tok["nm_q_diff"] == "") & (tok["nm_s_diff"] == "")
    nm_diff_ratio[both_empty.to_numpy()] = 100.0
    # are the query's extra address tokens approximately present in the S1 address (typos) or new?
    ad_qonly_in_s = _cp(fuzz.partial_ratio, tok["ad_q_diff"].to_list(), S["addr_norm"].to_list())
    ad_qonly_in_s[(tok["ad_q_diff"] == "").to_numpy()] = 100.0
    del qd, sd
    out = c.select("q_idx", "s1_idx", "cos_name", "cos_addr", "cos_comb", "rank", "n_cands",
                   "comb_max", "comb_gap", "comb_gap_next", "name_gap", "addr_gap").with_columns(
        [pl.Series(k, v) for k, v in f.items()]
    ).with_columns(
        (Q["name_core"] == S["name_core"]).alias("nm_exact"),
        (Q["name_core"].str.replace_all(" ", "") == S["name_core"].str.replace_all(" ", "")).alias("key_exact"),
        tok["tok_inter"],
        (tok["tok_inter"] / tok["tok_union"].clip(1)).alias("tok_jacc"),
        (tok["tok_inter"] / tok["q_ntok"].clip(1)).alias("tok_q_cov"),
        (tok["tok_inter"] / tok["s_ntok"].clip(1)).alias("tok_s_cov"),
        tok["first_tok_eq"],
        Q["name_core"].str.len_chars().alias("len_q"),
        S["name_core"].str.len_chars().alias("len_s"),
        (tok["ad_inter"] / tok["ad_union"].clip(1)).alias("ad_jacc"),
        (tok["ad_inter"] / tok["qa_n"].clip(1)).alias("ad_q_cov"),
        tok["num_q"], tok["num_s"], tok["num_inter"],
        (tok["num_inter"] / tok["num_q"].clip(1)).alias("num_q_cov"),
        tok["num_first_eq"],
        ((tok["num_q"] > 0) & (tok["num_s"] > 0) & (tok["num_inter"] == 0)).alias("num_conflict"),
        Q["addr_missing"].alias("q_addr_missing"),
        Q["addr_norm"].str.len_chars().alias("ad_len_q"),
        S["addr_norm"].str.len_chars().alias("ad_len_s"),
        Q["is_s3"].alias("q_is_s3"),
        Q["is_domain"].alias("q_domain"),
        Q["has_indic"].alias("q_indic"),
        (tok["nm_q_diff"].str.count_matches(r"\S+")).alias("nm_q_only"),
        (tok["nm_s_diff"].str.count_matches(r"\S+")).alias("nm_s_only"),
        pl.Series("nm_diff_ratio", nm_diff_ratio),
        (tok["ad_q_diff"].str.count_matches(r"\S+")).alias("ad_q_only"),
        pl.Series("ad_qonly_in_s", ad_qonly_in_s),
        tok["num_q_only"], tok["num_s_only"], tok["num_set_eq"],
        ((tok["qn1"] - tok["sn1"]).abs() / pl.max_horizontal(tok["qn1"], tok["sn1"], pl.lit(1.0))).fill_null(-1.0).alias("num_rel_diff"),
    ).with_columns(
        (pl.min_horizontal("len_q", "len_s") / pl.max_horizontal("len_q", "len_s").clip(1)).alias("len_ratio"),
    ).join(s1_extra, on="s1_idx", how="left")
    out = add_margins(out)
    return out.select(["q_idx", "s1_idx"] + [pl.col(x).cast(pl.Float32) for x in FEATURES])


def add_margins(d: pl.DataFrame) -> pl.DataFrame:
    """x - max(x over the *other* candidates of the query) for key similarity columns."""
    exprs = []
    for name, col in MARGIN_COLS.items():
        x = pl.col(col).cast(pl.Float32)
        mx1 = x.max().over("q_idx")
        mx2 = x.sort(descending=True).slice(1, 1).first().over("q_idx")
        exprs.append(pl.when(x == mx1).then(x - mx2).otherwise(x - mx1).fill_null(x).alias(name))
    ties = (pl.col("cos_name") >= 0.95 * pl.col("cos_name").max().over("q_idx")).sum().over("q_idx")
    return d.with_columns(exprs + [ties.cast(pl.Float32).alias("n_name_ties")])


def s1_extra_table(s1: pl.DataFrame, cand_dir) -> pl.DataFrame:
    """S1-level ambiguity/popularity: #S1 in same country with the same core name,
    #queries that retrieved this S1 at rank 1."""
    dup = s1.select("s1_idx", "country", "name_core").with_columns(
        pl.len().over("country", "name_core").alias("s1_name_dup")).select("s1_idx", "s1_name_dup")
    deg = (pl.scan_parquet(str(cand_dir / "*.parquet")).filter(pl.col("rank") == 1)
           .group_by("s1_idx").agg(pl.len().alias("s1_rank1_deg")).collect())
    return dup.join(deg, on="s1_idx", how="left").with_columns(
        pl.col("s1_rank1_deg").fill_null(0), pl.col("s1_idx").cast(pl.UInt32))

"""End-to-end pipeline: raw TSVs -> output/matching_results.tsv + output/candidate_pairs.tsv.

Each stage caches its result under cache/ and is skipped/resumed when the cache exists.
"""
import time

import apply_indic_dict
import build_features
import build_normalized
import candidates
import indic_dictionary
import predict
import prepare_data
import train
from config import cache_path


def main():
    t0 = time.time()
    if not cache_path("test_source3.parquet").exists():
        prepare_data.main()
    build_normalized.main(("train", "test"))
    # Indic-script names: learn token dictionary on the training fold, apply to both splits
    if not indic_dictionary.DICT_PATH.exists():
        indic_dictionary.learn()
    mapping = indic_dictionary.load()
    for split in ("train", "test"):
        apply_indic_dict.apply(split, mapping)
    for split in ("train", "test"):
        candidates.generate(split)
        build_features.build(split)
    if not train.MODEL_PATH.exists():
        train.main()
    predict.main()
    print(f"pipeline finished in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

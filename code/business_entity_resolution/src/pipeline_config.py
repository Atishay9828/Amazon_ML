"""Validate and fingerprint a reproducible Person A pipeline configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

try:
    from .candidates import Settings
    from .rank_candidates import FEATURE_NAMES
except ImportError:
    from candidates import Settings
    from rank_candidates import FEATURE_NAMES


TOP_LEVEL = {"reverse_top1", "candidates", "ranker", "pipeline_order",
             "expected_volume_test", "not_recommended"}
REVERSE_KEYS = {"enabled", "ngram", "hash_features", "df_cap_fraction", "top_k",
                "batch_size", "workers", "matrix_chunk", "index_source1_of_same_split"}
RANKER_KEYS = {"enabled", "train_sample_split", "train_source1_ids", "train_selection_rule",
               "model", "model_params", "max_negatives_per_source", "features",
               "country_policy", "final_cap", "final_cap_alternative", "selection_rule"}


def _without_notes(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _without_notes(item) for key, item in value.items()
                if not key.startswith("_")}
    if isinstance(value, list):
        return [_without_notes(item) for item in value]
    return value


def _check_keys(section: str, mapping: dict, allowed: set[str], required: set[str]) -> None:
    unknown = set(mapping) - allowed
    missing = required - set(mapping)
    if unknown or missing:
        raise ValueError(f"{section}: unknown keys {sorted(unknown)}; missing keys {sorted(missing)}")


def _digest(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PipelineConfig:
    values: dict[str, Any]
    sha256: str

    @property
    def reverse_sha256(self) -> str:
        """Reverse files depend only on their own section, so config edits elsewhere keep them valid."""
        return _digest(self.values["reverse_top1"])

    @property
    def reverse_enabled(self) -> bool:
        return self.values["reverse_top1"].get("enabled", True)

    @property
    def ranker_enabled(self) -> bool:
        return self.values["ranker"].get("enabled", True)

    def settings(self, reverse_pairs: Path | None = None) -> Settings:
        values = dict(self.values["candidates"])
        declared = values.pop("extra_pairs")
        if self.reverse_enabled:
            if declared != "<output of step 1 for the same split>" or reverse_pairs is None:
                raise ValueError("enabled reverse channel requires its same-split pair file")
        elif declared is not None or reverse_pairs is not None:
            raise ValueError("disabled reverse channel cannot receive extra pairs")
        return Settings(**values, extra_pairs=reverse_pairs, config_hash=self.sha256)


def load_config(path: Path) -> PipelineConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("config must be a JSON object")
    values = _without_notes(raw)
    _check_keys("config", values, TOP_LEVEL, {"reverse_top1", "candidates", "ranker"})
    reverse = values["reverse_top1"]
    candidate = values["candidates"]
    ranker = values["ranker"]
    if not all(isinstance(section, dict) for section in (reverse, candidate, ranker)):
        raise ValueError("pipeline sections must be JSON objects")
    _check_keys("reverse_top1", reverse, REVERSE_KEYS,
                {"enabled"} if reverse.get("enabled") is False else
                REVERSE_KEYS - {"enabled"})
    candidate_fields = {field.name for field in fields(Settings)} - {"config_hash"}
    _check_keys("candidates", candidate, candidate_fields, candidate_fields)
    _check_keys("ranker", ranker, RANKER_KEYS,
                {"enabled", "final_cap"} if ranker.get("enabled") is False else
                RANKER_KEYS - {"enabled", "final_cap_alternative"})
    if reverse.get("enabled", True):
        if (reverse["top_k"] != 1 or not reverse["index_source1_of_same_split"] or
                reverse["ngram"] != "char 3-gram on normalize_name + ' | ' + normalize_address"):
            raise ValueError("unsupported reverse search settings")
    if candidate["sibling_seeds"] and candidate["hop_seeds"]:
        raise ValueError("enable only one sibling retrieval mechanism")
    if ranker.get("enabled", True):
        if ranker["train_sample_split"] != "ranktrain" or ranker["train_source1_ids"] != 50_000:
            raise ValueError("ranker needs the disjoint 50,000-record ranktrain sample")
        if ranker["model"] != "sklearn.ensemble.HistGradientBoostingClassifier":
            raise ValueError("unsupported ranker model")
        _check_keys("ranker.model_params", ranker["model_params"],
                    {"max_iter", "learning_rate", "max_leaf_nodes", "random_state"},
                    {"max_iter", "learning_rate", "max_leaf_nodes", "random_state"})
        if ranker["final_cap"] not in {16, 32}:
            raise ValueError("learned final cap must be 16 or 32")
        feature_names = [name for name in ranker["features"] if name != "ch_<every channel in CHANNEL_ORDER>"]
        feature_names += [name for name in FEATURE_NAMES if name.startswith("ch_")]
        if tuple(feature_names) != FEATURE_NAMES:
            raise ValueError("ranker feature list differs from the implementation")
        if "feature only" not in ranker["country_policy"]:
            raise ValueError("country must remain a feature, never a retrieval filter")
        if "cap // 2 per source" not in ranker["selection_rule"]:
            raise ValueError("unsupported ranker selection rule")
    return PipelineConfig(values, _digest(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(load_config(args.config).sha256)


if __name__ == "__main__":
    main()

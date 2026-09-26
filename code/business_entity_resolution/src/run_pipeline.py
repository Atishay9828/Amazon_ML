"""Run the configured Person A candidate pipeline with resumable checkpoints."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import time
from pathlib import Path

try:
    from . import candidates, rank_candidates, reverse_top1
    from .assemble_staged import assemble
    from .data import sha256_file, source_path
    from .evaluate_staged import evaluate as evaluate_staged
    from .merge_candidate_shards import merge
    from .pipeline_config import PipelineConfig, load_config
except ImportError:
    import candidates
    import rank_candidates
    import reverse_top1
    from assemble_staged import assemble
    from data import sha256_file, source_path
    from evaluate_staged import evaluate as evaluate_staged
    from merge_candidate_shards import merge
    from pipeline_config import PipelineConfig, load_config


def _reverse(data_root: Path, split: str, work_dir: Path,
             config: PipelineConfig, supplied: Path | None) -> Path | None:
    if not config.reverse_enabled:
        if supplied:
            raise ValueError("reverse file supplied for a config without reverse search")
        return None
    path = supplied or work_dir / f"reverse-{split}.tsv"
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    settings = config.values["reverse_top1"]
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        expected = {"split": split, "source1_sha256": sha256_file(source_path(data_root, split, 1)),
                    "limit_targets": None,
                    "hash_features": settings["hash_features"],
                    "df_cap_fraction": settings["df_cap_fraction"],
                    "batch_size": settings["batch_size"]}
        if all(meta.get(key) == value for key, value in expected.items()):
            print(json.dumps({"stage": "reverse_reused", "path": str(path)}), flush=True)
            return path
        raise ValueError(f"{path}: reverse metadata does not match this config and split")
    if supplied or path.exists() or meta_path.exists():
        raise ValueError(f"{path}: incomplete reverse output or missing metadata")
    reverse_top1.run(data_root, split, path, workers=settings["workers"],
                     batch_size=settings["batch_size"],
                     hash_features=settings["hash_features"],
                     df_cap_fraction=settings["df_cap_fraction"],
                     matrix_chunk=settings["matrix_chunk"], config_hash=config.reverse_sha256)
    return path


def _reverse_blank(data_root: Path, split: str, work_dir: Path,
                   config: PipelineConfig) -> Path | None:
    if not config.blank_enabled:
        return None
    path = work_dir / f"reverse-blank-{split}.tsv"
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    settings = config.values["reverse_blank_name"]
    expected = {"split": split, "channel": "reverse_blank_name",
                "source1_sha256": sha256_file(source_path(data_root, split, 1)),
                "limit_targets": None, "top_k": settings["top_k"],
                "hash_features": settings["hash_features"],
                "df_cap_fraction": settings["df_cap_fraction"], "batch_size": settings["batch_size"]}
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if all(meta.get(key) == value for key, value in expected.items()):
            print(json.dumps({"stage": "reverse_blank_reused", "path": str(path)}), flush=True)
            return path
        raise ValueError(f"{path}: blank-address reverse metadata does not match this config and split")
    reverse_top1.run_blank_name(data_root, split, path, top_k=settings["top_k"],
                                workers=settings["workers"], batch_size=settings["batch_size"],
                                hash_features=settings["hash_features"],
                                df_cap_fraction=settings["df_cap_fraction"],
                                matrix_chunk=settings["matrix_chunk"], config_hash=config.blank_sha256)
    return path


def _stage(data_root: Path, split: str, sample_split: str | None,
           work_dir: Path, config: PipelineConfig, reverse_path: Path | None,
           shard_index: int = 0, shard_count: int = 1,
           blank_path: Path | None = None) -> tuple[Path, Path]:
    suffix = sample_split or (f"{split}-{shard_index}-of-{shard_count}"
                              if shard_count > 1 else split)
    stage_dir = work_dir / f"stage-{suffix}"
    interim = work_dir / f"interim-{suffix}.tsv"
    settings = config.settings(reverse_path, blank_path)
    expected = candidates._manifest(data_root, split, settings, None, None,
                                    sample_split, shard_index, shard_count)
    manifest_path = stage_dir / "manifest.json"
    if interim.is_file() and manifest_path.is_file():
        found = json.loads(manifest_path.read_text(encoding="utf-8"))
        if found == expected:
            print(json.dumps({"stage": "candidates_reused", "path": str(interim)}), flush=True)
            return stage_dir, interim
    candidates.generate(data_root, split, interim, stage_dir, settings,
                        sample_split=sample_split, shard_index=shard_index,
                        shard_count=shard_count)
    return stage_dir, interim


def _model(data_root: Path, work_dir: Path, config: PipelineConfig,
           supplied: Path | None, train_reverse: Path | None) -> Path:
    path = supplied or work_dir / f"ranker-{config.sha256[:16]}.pkl"
    ranker_config = config.values["ranker"]
    if path.is_file():
        with path.open("rb") as handle:
            bundle = pickle.load(handle)
        if (bundle.get("config_hash") != config.sha256 or
                tuple(bundle.get("feature_names", ())) != rank_candidates.FEATURE_NAMES or
                bundle.get("model_params") != ranker_config["model_params"] or
                bundle.get("max_negatives_per_source") != ranker_config["max_negatives_per_source"]):
            raise ValueError(f"{path}: ranker model does not match this config")
        print(json.dumps({"stage": "ranker_reused", "path": str(path)}), flush=True)
        return path
    if supplied:
        raise FileNotFoundError(f"supplied ranker model does not exist: {path}")
    stage_dir, _ = _stage(data_root, "train", "ranktrain", work_dir,
                          config, train_reverse,
                          blank_path=_reverse_blank(data_root, "train", work_dir, config))
    rank_candidates.train(data_root, stage_dir, path,
                          max_negatives_per_source=ranker_config["max_negatives_per_source"],
                          model_params=ranker_config["model_params"])
    return path


def _output_metadata(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".meta.json")


def _inference(data_root: Path, split: str, sample_split: str | None,
               stage_dir: Path, model: Path, out: Path, cap: int,
               config: PipelineConfig, shard_index: int, shard_count: int) -> dict:
    meta_path = _output_metadata(out)
    expected = {"config_hash": config.sha256, "split": split,
                "sample_split": sample_split, "shard_index": shard_index,
                "shard_count": shard_count, "cap": cap,
                "model_sha256": sha256_file(model)}
    if out.is_file() and meta_path.is_file():
        found = json.loads(meta_path.read_text(encoding="utf-8"))
        if all(found.get(key) == value for key, value in expected.items()):
            print(json.dumps({"stage": "inference_reused", "path": str(out)}), flush=True)
            return found
    report = rank_candidates.infer(data_root, split, stage_dir, model, out, cap,
                                   sample_split, shard_index, shard_count)
    metadata = {**expected, **report}
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def run(config_path: Path, data_root: Path, split: str, work_dir: Path, out: Path,
        *, sample_split: str | None = None, shard_index: int = 0,
        shard_count: int = 1, reverse_train: Path | None = None,
        reverse_test: Path | None = None, model_path: Path | None = None,
        dev_report: bool = True) -> dict:
    if split not in {"train", "test"} or (sample_split and split != "train"):
        raise ValueError("sample splits require the train split")
    if shard_count < 1 or not 0 <= shard_index < shard_count or (sample_split and shard_count != 1):
        raise ValueError("invalid shard selection")
    started = time.monotonic()
    config = load_config(config_path)
    work_dir.mkdir(parents=True, exist_ok=True)
    if not config.ranker_enabled:
        reverse_path = _reverse(data_root, split, work_dir, config,
                                reverse_train if split == "train" else reverse_test)
        stage_dir, interim = _stage(data_root, split, sample_split, work_dir,
                                    config, reverse_path, shard_index, shard_count,
                                    _reverse_blank(data_root, split, work_dir, config))
        if interim != out:
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(interim, out)
        result = {"config_hash": config.sha256, "split": split, "sample_split": sample_split,
                  "stage_dir": str(stage_dir), "out": str(out), "model": None}
    else:
        if model_path is None:
            train_reverse = _reverse(data_root, "train", work_dir, config, reverse_train)
        else:
            train_reverse = None
        model = _model(data_root, work_dir, config, model_path, train_reverse)
        reverse_path = (train_reverse if split == "train" else
                        _reverse(data_root, "test", work_dir, config, reverse_test))
        if split == "train" and reverse_path is None and config.reverse_enabled:
            reverse_path = _reverse(data_root, "train", work_dir, config, reverse_train)
        stage_dir, _ = _stage(data_root, split, sample_split, work_dir,
                              config, reverse_path, shard_index, shard_count,
                              _reverse_blank(data_root, split, work_dir, config))
        cap = config.values["ranker"]["final_cap"]
        out.parent.mkdir(parents=True, exist_ok=True)
        inferred = _inference(data_root, split, sample_split, stage_dir,
                              model, out, cap, config, shard_index, shard_count)
        result = {"config_hash": config.sha256, "split": split, "sample_split": sample_split,
                  "stage_dir": str(stage_dir), "out": str(out), "model": str(model),
                  "inference": inferred}
        if sample_split == "dev" and dev_report:
            selected = {record.entity_id for record in candidates._select_source1(
                candidates._load_sorted_source1(source_path(data_root, "train", 1), None), "dev")}
            truth_path = data_root / "train" / "train_ground_truth.tsv"
            metrics = {"staged": evaluate_staged(stage_dir, data_root, "dev")}
            for variant_cap in (16, 32, 64):
                learned_path = (out if variant_cap == cap else
                                out.with_name(f"{out.stem}-cap{variant_cap}.tsv"))
                _inference(data_root, split, sample_split, stage_dir, model,
                           learned_path, variant_cap, config, shard_index, shard_count)
                metrics[f"learned_cap{variant_cap}"] = candidates.evaluate_retrieval(
                    learned_path, truth_path, selected)
                control_path = out.with_name(f"{out.stem}-control-cap{variant_cap}.tsv")
                assemble(data_root, split, stage_dir, control_path, variant_cap,
                         sample_split=sample_split, final_score="current")
                metrics[f"control_cap{variant_cap}"] = candidates.evaluate_retrieval(
                    control_path, truth_path, selected)
            result["dev_metrics"] = metrics
            report_path = work_dir / "dev-report.json"
            report_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    result["seconds"] = round(time.monotonic() - started, 2)
    result.update(candidates._memory_state())
    print(json.dumps({"stage": "pipeline_complete", **result}), flush=True)
    return result


def merge_shards(config_path: Path, split: str, work_dir: Path,
                 out: Path, shard_count: int) -> dict:
    if shard_count < 2:
        raise ValueError("merge needs at least two shards")
    config = load_config(config_path)
    paths = [work_dir / "results" / f"{split}-{index}-of-{shard_count}.tsv"
             for index in range(shard_count)]
    for index, path in enumerate(paths):
        metadata = json.loads(_output_metadata(path).read_text(encoding="utf-8"))
        if (metadata.get("config_hash") != config.sha256 or metadata.get("split") != split or
                metadata.get("shard_index") != index or metadata.get("shard_count") != shard_count):
            raise ValueError(f"shard metadata differs from config: {path}")
    count = merge(paths, out)
    report = {"config_hash": config.sha256, "split": split, "shard_count": shard_count,
              "candidate_pairs": count}
    _output_metadata(out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "shards_merged", **report}), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--split", choices=("train", "test"))
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sample-split", choices=("dev", "holdout"))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--reverse-train", type=Path,
                        help="Private precomputed reverse file with matching metadata")
    parser.add_argument("--reverse-test", type=Path,
                        help="Private precomputed reverse file with matching metadata")
    parser.add_argument("--model", type=Path,
                        help="Private precomputed ranker model with matching config hash")
    parser.add_argument("--merge-shards", action="store_true")
    args = parser.parse_args()
    if args.merge_shards:
        if args.data_root or args.sample_split or args.model or args.reverse_train or args.reverse_test:
            parser.error("merge uses only config, split, work-dir, out, and shard-count")
        if not args.split:
            parser.error("--split is required")
        merge_shards(args.config, args.split, args.work_dir, args.out, args.shard_count)
    else:
        if not args.data_root or not args.split:
            parser.error("--data-root and --split are required")
        run(args.config, args.data_root, args.split, args.work_dir, args.out,
            sample_split=args.sample_split, shard_index=args.shard_index,
            shard_count=args.shard_count, reverse_train=args.reverse_train,
            reverse_test=args.reverse_test, model_path=args.model)


if __name__ == "__main__":
    main()

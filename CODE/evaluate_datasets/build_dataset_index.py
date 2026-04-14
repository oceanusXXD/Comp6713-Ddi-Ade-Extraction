#!/usr/bin/env python3
"""Build evaluation dataset index files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = REPO_ROOT / "evaluate_datasets"
REPORT_STATS = REPO_ROOT / "reports" / "augment" / "data_stats.json"
FALLBACK_STATS = REPO_ROOT / "reports" / "data_stats.json"
OUTPUT_JSON = EVAL_ROOT / "DATASET_INDEX.json"
OUTPUT_MD = EVAL_ROOT / "DATASET_INDEX.md"


def ensure_parent(path: Path) -> None:
    """Ensure the parent directory for an output file exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Dict[str, Any]:
    """Read a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def count_jsonl_rows(path: Path) -> int | None:
    """Count JSONL rows and return `None` when the file is missing."""
    if not path.exists():
        return None
    return sum(1 for _ in path.open("r", encoding="utf-8"))


def build_index() -> Dict[str, Any]:
    """Collect internal and external evaluation datasets into one index."""
    if REPORT_STATS.exists():
        stats = read_json(REPORT_STATS)
    elif FALLBACK_STATS.exists():
        stats = read_json(FALLBACK_STATS)
    else:
        stats = {}
    manifest_path = EVAL_ROOT / "MANIFEST.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}

    train_path = REPO_ROOT / "data" / "processed" / "Comp6713-Ddi-Ade-Extraction_final_augment" / "merged_chatml_train.jsonl"
    validation_path = REPO_ROOT / "data" / "processed" / "Comp6713-Ddi-Ade-Extraction_final_augment" / "merged_chatml_validation.jsonl"
    test_path = REPO_ROOT / "data" / "processed" / "Comp6713-Ddi-Ade-Extraction_final_augment" / "merged_chatml_test.jsonl"
    augment_path = REPO_ROOT / "data" / "processed" / "Comp6713-Ddi-Ade-Extraction_final_augment" / "merged_chatml_train_augmentations.jsonl"

    train_rows = count_jsonl_rows(train_path)
    validation_rows = count_jsonl_rows(validation_path)
    test_rows = count_jsonl_rows(test_path)
    augment_rows = count_jsonl_rows(augment_path)
    augment_type_counts = (
        stats.get("files", {})
        .get("final_augmentations", {})
        .get("augmentation_type_counts", {})
    )

    index = {
        "internal": {
            "train": {
                "path": str(train_path),
                "num_rows": stats.get("files", {}).get("final_train", {}).get("num_rows", train_rows),
            },
            "validation": {
                "path": str(validation_path),
                "num_rows": stats.get("files", {}).get("original_validation", {}).get("num_rows", validation_rows),
            },
            "test": {
                "path": str(test_path),
                "num_rows": stats.get("files", {}).get("original_test", {}).get("num_rows", test_rows),
            },
            "augmentations": {
                "path": str(augment_path),
                "num_rows": stats.get("files", {}).get("final_augmentations", {}).get("num_rows", augment_rows),
                "augmentation_type_counts": augment_type_counts,
            },
        },
        "external": {
            "seen_style_core": {
                "readme": str(EVAL_ROOT / "seen_style_core" / "README.md"),
                "validation_path": str(EVAL_ROOT / "seen_style_core" / "official_held_out" / "merged_chatml_validation.jsonl"),
                "test_path": str(EVAL_ROOT / "seen_style_core" / "official_held_out" / "merged_chatml_test.jsonl"),
                "validation_rows": count_jsonl_rows(EVAL_ROOT / "seen_style_core" / "official_held_out" / "merged_chatml_validation.jsonl"),
                "test_rows": count_jsonl_rows(EVAL_ROOT / "seen_style_core" / "official_held_out" / "merged_chatml_test.jsonl"),
                "used_by_default_training": False,
                "used_by_default_validation": False,
                "used_as_optional_same_style_eval": True,
            },
            "bundles": manifest,
        },
    }
    return index


def build_markdown(index: Dict[str, Any]) -> str:
    """Convert the index structure into Markdown."""
    internal = index["internal"]
    external = index["external"]
    lines = [
        "# Dataset Index",
        "",
        "## Current internal train / validation / test data",
        "",
        f"- Train data: `{internal['train']['path']}`, `{internal['train']['num_rows']}` rows",
        f"- Validation data: `{internal['validation']['path']}`, `{internal['validation']['num_rows']}` rows",
        f"- Test data: `{internal['test']['path']}`, `{internal['test']['num_rows']}` rows",
        f"- Augmentation sidecar: `{internal['augmentations']['path']}`, `{internal['augmentations']['num_rows']}` rows",
        f"- Augmentation type counts: `{json.dumps(internal['augmentations']['augmentation_type_counts'], ensure_ascii=False)}`",
        "",
        "## Current external evaluation data",
        "",
        f"- Same-style held-out validation: `{external['seen_style_core']['validation_path']}`, `{external['seen_style_core']['validation_rows']}` rows",
        f"- Same-style held-out test: `{external['seen_style_core']['test_path']}`, `{external['seen_style_core']['test_rows']}` rows",
        "- `seen_style_core` is not part of the default training or validation setup and is kept as an optional same-style evaluation entry.",
        "",
        "## External bundle directories",
        "",
    ]

    bundles = external["bundles"]
    for bundle_name in sorted(bundles):
        bundle = bundles[bundle_name]
        status = bundle.get("status", "unknown")
        lines.append(f"- `{bundle_name}`: status `{status}`")

    lines.extend(
        [
            "",
            "## Usage",
            "",
            "- Redownload and reorganize external evaluation datasets: `bash evaluate_datasets/download_evaluate_datasets.sh`",
            "- Rebuild the index: `python evaluate_datasets/build_dataset_index.py`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Write both JSON and Markdown index files."""
    index = build_index()
    ensure_parent(OUTPUT_JSON)
    OUTPUT_JSON.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUTPUT_MD.write_text(build_markdown(index), encoding="utf-8")
    print(f"Saved JSON index to: {OUTPUT_JSON}")
    print(f"Saved Markdown index to: {OUTPUT_MD}")


if __name__ == "__main__":
    main()

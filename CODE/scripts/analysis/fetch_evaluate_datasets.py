#!/usr/bin/env python3
"""External evaluation dataset fetch and organization script."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from huggingface_hub import hf_hub_download


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "evaluate_datasets"
CHUNK_SIZE = 1024 * 1024
LONGBENCH_LIGHT_TASKS = {
    "multifieldqa_en": 25,
    "multifieldqa_zh": 25,
    "passage_retrieval_en": 25,
    "passage_retrieval_zh": 25,
    "gov_report": 25,
    "vcsum": 25,
}


def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_text(path: Path, text: str) -> None:
    """Write a text file and create parent directories automatically."""
    ensure_dir(path.parent)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def jsonable(value: Any) -> Any:
    """Convert numpy/pandas scalars and similar objects to JSON-serializable plain types."""
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def download_file(url: str, destination: Path, force: bool = False) -> Path:
    """Download a file from a URL."""
    ensure_dir(destination.parent)
    if destination.exists() and not force and destination.stat().st_size > 0:
        print(f"[skip] {destination}")
        return destination

    print(f"[download] {url} -> {destination}")
    with requests.get(
        url,
        stream=True,
        timeout=120,
        headers={"User-Agent": "Comp6713-Ddi-Ade-Extraction dataset fetcher"},
    ) as response:
        response.raise_for_status()
        tmp_path = destination.with_suffix(destination.suffix + ".part")
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    handle.write(chunk)
        tmp_path.replace(destination)
    return destination


def extract_archive(archive_path: Path, destination: Path, force: bool = False) -> Path:
    """Extract a zip/tar.gz archive and use a marker file to skip repeat extraction."""
    marker = destination / ".extracted"
    if marker.exists() and not force:
        print(f"[skip] extracted {destination}")
        return destination

    if destination.exists() and force:
        shutil.rmtree(destination)
    ensure_dir(destination)

    print(f"[extract] {archive_path} -> {destination}")
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(destination)
    elif archive_path.suffixes[-2:] == [".tar", ".gz"] or archive_path.suffix == ".tgz":
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(destination)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")

    marker.touch()
    return destination


def download_hf_file(
    repo_id: str,
    filename: str,
    destination: Path,
    repo_type: str = "dataset",
    force: bool = False,
) -> Path:
    if destination.exists() and not force and destination.stat().st_size > 0:
        print(f"[skip] {destination}")
        return destination

    print(f"[hf] {repo_id}:{filename} -> {destination}")
    cached = Path(hf_hub_download(repo_id=repo_id, repo_type=repo_type, filename=filename))
    ensure_dir(destination.parent)
    shutil.copy2(cached, destination)
    return destination


def export_parquet_to_jsonl(parquet_path: Path, jsonl_path: Path) -> None:
    if jsonl_path.exists():
        print(f"[skip] {jsonl_path}")
        return

    print(f"[export] {parquet_path} -> {jsonl_path}")
    ensure_dir(jsonl_path.parent)
    dataframe = pd.read_parquet(parquet_path)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in dataframe.to_dict(orient="records"):
            handle.write(json.dumps(jsonable(record), ensure_ascii=False) + "\n")


def symlink_or_copy(source: Path, destination: Path) -> None:
    ensure_dir(destination.parent)
    if destination.exists() or destination.is_symlink():
        return

    relative_source = Path(os.path.relpath(source, destination.parent))
    try:
        destination.symlink_to(relative_source)
    except OSError:
        shutil.copy2(source, destination)


def fetch_cadec_v2(destination: Path, force: bool = False) -> tuple[Path, str]:
    api_url = "https://data.csiro.au/dap/ws/v2/collections/17190/data"
    metadata = requests.get(api_url, timeout=30).json()
    for file_info in metadata["file"]:
        if file_info["filename"] == "CADEC.v2.zip":
            source_url = file_info["presignedLink"]["href"]
            return download_file(source_url, destination, force=force), api_url
    raise RuntimeError("Unable to find CADEC.v2.zip in CSIRO metadata")


def build_seen_style_core(output_dir: Path, manifest: dict[str, Any]) -> None:
    """Build the held-out same-style evaluation mirror."""
    bundle_dir = ensure_dir(output_dir / "seen_style_core" / "official_held_out")
    validation_source = REPO_ROOT / "data" / "merged_chatml_validation.jsonl"
    test_source = REPO_ROOT / "data" / "merged_chatml_test.jsonl"
    symlink_or_copy(validation_source, bundle_dir / "merged_chatml_validation.jsonl")
    symlink_or_copy(test_source, bundle_dir / "merged_chatml_test.jsonl")

    readme = """# Seen-Style Core Evaluation Set

This directory points to the repository's current held-out same-style files:

- `official_held_out/merged_chatml_validation.jsonl`
- `official_held_out/merged_chatml_test.jsonl`

The requested `hard`, `negative`, and `schema_stability` subsets are not materialized separately because the repository does not define a reproducible split rule or provide enough source metadata to derive them without guesswork.
"""
    write_text(output_dir / "seen_style_core" / "README.md", readme)

    manifest["seen_style_core"] = {
        "status": "partial",
        "official_held_out": [
            str(bundle_dir / "merged_chatml_validation.jsonl"),
            str(bundle_dir / "merged_chatml_test.jsonl"),
        ],
        "pending_subsets": ["hard", "negative", "schema_stability"],
        "note": "The repository only provides the held-out files and does not define reproducible split rules for these three internal subsets.",
    }


def build_ddi_transfer(output_dir: Path, manifest: dict[str, Any], force: bool) -> None:
    bundle_dir = ensure_dir(output_dir / "ddi_transfer")
    ddi2013_dir = ensure_dir(bundle_dir / "DDIExtraction2013")
    ddi2013_raw = ddi2013_dir / "raw" / "DDICorpus-2013.zip"
    ddi2013_source = "https://raw.githubusercontent.com/isegura/DDICorpus/master/DDICorpus-2013.zip"
    download_file(ddi2013_source, ddi2013_raw, force=force)
    extract_archive(ddi2013_raw, ddi2013_dir / "extracted", force=force)

    tac2018_dir = ensure_dir(bundle_dir / "TAC2018_DDI")
    tac2018_sources = {
        "trainingFiles.zip": "https://bionlp.nlm.nih.gov/tac2018druginteractions/trainingFiles.zip",
        "test1Files.zip": "https://bionlp.nlm.nih.gov/tac2018druginteractions/test1Files.zip",
        "test2Files.zip": "https://bionlp.nlm.nih.gov/tac2018druginteractions/test2Files.zip",
    }
    for filename, url in tac2018_sources.items():
        archive_path = tac2018_dir / "raw" / filename
        download_file(url, archive_path, force=force)
        extract_archive(archive_path, tac2018_dir / "extracted" / filename.removesuffix(".zip"), force=force)

    write_text(
        bundle_dir / "README.md",
        """# DDI Transfer Evaluation Bundle

- `DDIExtraction2013/` contains the public `DDIExtraction 2013` archive and its extracted contents.
- `TAC2018_DDI/` contains the public `TAC 2018 DDI` train and test archives and their extracted contents.
""",
    )

    manifest["ddi_transfer"] = {
        "status": "complete",
        "datasets": {
            "DDIExtraction2013": ddi2013_source,
            "TAC2018_DDI": tac2018_sources,
        },
    }


def build_ade_transfer(output_dir: Path, manifest: dict[str, Any], force: bool) -> None:
    bundle_dir = ensure_dir(output_dir / "ade_transfer")

    ade_dir = ensure_dir(bundle_dir / "ADE_Corpus_V2")
    ade_parquet = ade_dir / "raw" / "Ade_corpus_v2_drug_ade_relation.parquet"
    download_hf_file(
        repo_id="ade-benchmark-corpus/ade_corpus_v2",
        filename="Ade_corpus_v2_drug_ade_relation/train-00000-of-00001.parquet",
        destination=ade_parquet,
        force=force,
    )
    export_parquet_to_jsonl(ade_parquet, ade_dir / "processed" / "drug_ade_relation.jsonl")

    phee_dir = ensure_dir(bundle_dir / "PHEE")
    phee_files = {
        "train.json": "https://raw.githubusercontent.com/ZhaoyueSun/PHEE/master/data/json/train.json",
        "dev.json": "https://raw.githubusercontent.com/ZhaoyueSun/PHEE/master/data/json/dev.json",
        "test.json": "https://raw.githubusercontent.com/ZhaoyueSun/PHEE/master/data/json/test.json",
    }
    for filename, url in phee_files.items():
        download_file(url, phee_dir / "raw" / filename, force=force)

    write_text(
        bundle_dir / "README.md",
        """# ADE Transfer Evaluation Bundle

- `ADE_Corpus_V2/processed/drug_ade_relation.jsonl` is exported from the public Hugging Face parquet file and corresponds to the binary drug-AE relation subset.
- `PHEE/raw/` stores the official `train/dev/test` JSON splits from the public `PHEE` repository.

Note: the `PHEE` README also points to a later `PHEE v2.0` repository. This bundle keeps the dataset variant explicitly included here.
""",
    )

    manifest["ade_transfer"] = {
        "status": "complete",
        "datasets": {
            "ADE_Corpus_V2": "https://huggingface.co/datasets/ade-benchmark-corpus/ade_corpus_v2",
            "PHEE": phee_files,
            "PHEE_v2_note": "https://github.com/ZhaoyueSun/phee-with-chatgpt",
        },
    }


def build_pharmacovigilance_cross_genre(output_dir: Path, manifest: dict[str, Any], force: bool) -> None:
    bundle_dir = ensure_dir(output_dir / "pharmacovigilance_cross_genre")

    tac2017_dir = ensure_dir(bundle_dir / "TAC2017_ADR")
    tac2017_sources = {
        "train_xml.tar.gz": "https://bionlp.nlm.nih.gov/tac2017adversereactions/train_xml.tar.gz",
        "gold_xml.tar.gz": "https://bionlp.nlm.nih.gov/tac2017adversereactions/gold_xml.tar.gz",
        "unannotated_xml.tar.gz": "https://bionlp.nlm.nih.gov/tac2017adversereactions/unannotated_xml.tar.gz",
    }
    for filename, url in tac2017_sources.items():
        archive_path = tac2017_dir / "raw" / filename
        download_file(url, archive_path, force=force)
        extract_archive(archive_path, tac2017_dir / "extracted" / filename.removesuffix(".tar.gz"), force=force)

    cadec_dir = ensure_dir(bundle_dir / "CADEC")
    cadec_archive, cadec_api_url = fetch_cadec_v2(cadec_dir / "raw" / "CADEC.v2.zip", force=force)
    extract_archive(cadec_archive, cadec_dir / "extracted", force=force)

    write_text(
        bundle_dir / "README.md",
        """# Cross-Genre Pharmacovigilance Evaluation Bundle

- `TAC2017_ADR/` contains the public `TAC 2017 ADR` training, gold, and unannotated XML archives.
- `CADEC/` contains the official `CADEC v2` archive fetched from the public CSIRO metadata endpoint and its extracted contents.
""",
    )

    manifest["pharmacovigilance_cross_genre"] = {
        "status": "complete",
        "datasets": {
            "TAC2017_ADR": tac2017_sources,
            "CADEC": cadec_api_url,
        },
    }


def build_longbench_light(raw_zip_path: Path, destination: Path) -> dict[str, int]:
    ensure_dir(destination)
    counts: dict[str, int] = {}
    with zipfile.ZipFile(raw_zip_path) as archive:
        for task_name, max_examples in LONGBENCH_LIGHT_TASKS.items():
            member_name = f"data/{task_name}.jsonl"
            output_path = destination / f"{task_name}.jsonl"
            written = 0
            with archive.open(member_name) as input_handle, output_path.open("w", encoding="utf-8") as output_handle:
                for raw_line in input_handle:
                    output_handle.write(raw_line.decode("utf-8"))
                    written += 1
                    if written >= max_examples:
                        break
            counts[task_name] = written
    return counts


def build_general_guardrails(output_dir: Path, manifest: dict[str, Any], force: bool) -> None:
    bundle_dir = ensure_dir(output_dir / "general_guardrails")

    ifeval_dir = ensure_dir(bundle_dir / "IFEval")
    ifeval_path = ifeval_dir / "raw" / "ifeval_input_data.jsonl"
    download_hf_file(
        repo_id="google/IFEval",
        filename="ifeval_input_data.jsonl",
        destination=ifeval_path,
        force=force,
    )

    docred_dir = ensure_dir(bundle_dir / "DocRED")
    docred_files = [
        "data/train_annotated.json.gz",
        "data/dev.json.gz",
        "data/test.json.gz",
        "data/rel_info.json.gz",
    ]
    for filename in docred_files:
        destination = docred_dir / "raw" / Path(filename).name
        download_hf_file(
            repo_id="thunlp/docred",
            filename=filename,
            destination=destination,
            force=force,
        )

    longbench_dir = ensure_dir(bundle_dir / "LongBench")
    longbench_zip = longbench_dir / "raw" / "data.zip"
    download_hf_file(
        repo_id="zai-org/LongBench",
        filename="data.zip",
        destination=longbench_zip,
        force=force,
    )
    counts = build_longbench_light(longbench_zip, longbench_dir / "light")
    write_text(
        longbench_dir / "light" / "README.md",
        """# LongBench Light Subset

This light subset keeps 6 `LongBench` tasks with 25 examples per task:

- `multifieldqa_en`
- `multifieldqa_zh`
- `passage_retrieval_en`
- `passage_retrieval_zh`
- `gov_report`
- `vcsum`

It preserves long-context QA, retrieval, summarization, and English/Chinese coverage without unpacking the full benchmark.
""",
    )

    manifest["general_guardrails"] = {
        "status": "complete",
        "datasets": {
            "IFEval": "https://huggingface.co/datasets/google/IFEval",
            "DocRED": "https://huggingface.co/datasets/thunlp/docred",
            "LongBench_full_raw": "https://huggingface.co/datasets/zai-org/LongBench",
            "LongBench_light_counts": counts,
        },
    }


def write_root_readme(output_dir: Path) -> None:
    """Write the root README for `evaluate_datasets/`."""
    readme = """# Evaluation Datasets

This directory stores the external evaluation bundles used by the project and their notes.

## Subdirectories

- `seen_style_core/`
  Points to the repository's held-out ChatML validation and test sets and notes the internal subsets that are not materialized here.
- `ddi_transfer/`
  Contains the public `DDIExtraction 2013` and `TAC 2018 DDI` bundles.
- `ade_transfer/`
  Contains `ADE-Corpus-V2` drug-AE relation data and the public `PHEE` split files.
- `pharmacovigilance_cross_genre/`
  Contains `TAC 2017 ADR` and `CADEC v2`.
- `general_guardrails/`
  Contains `IFEval`, `DocRED`, and the reduced `LongBench light` subset.

## Download and Processing

Run from the repository root:

```bash
python scripts/analysis/fetch_evaluate_datasets.py
```

The script skips existing files. Use `--force` to redownload and re-extract.
"""
    write_text(output_dir / "README.md", readme)


def main() -> None:
    """Fetch and organize all configured evaluation bundles."""
    parser = argparse.ArgumentParser(description="Download and organize evaluation datasets under evaluate_datasets/")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for evaluation bundles.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files and re-extract archives.",
    )
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir.resolve())
    manifest: dict[str, Any] = {}

    build_seen_style_core(output_dir, manifest)
    build_ddi_transfer(output_dir, manifest, force=args.force)
    build_ade_transfer(output_dir, manifest, force=args.force)
    build_pharmacovigilance_cross_genre(output_dir, manifest, force=args.force)
    build_general_guardrails(output_dir, manifest, force=args.force)
    write_root_readme(output_dir)
    write_text(output_dir / "MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))

    print(f"[done] wrote datasets to {output_dir}")


if __name__ == "__main__":
    main()

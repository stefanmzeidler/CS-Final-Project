import os
from typing import Any

from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from pathlib import Path
import json
import torch


def xml_to_arrow(source: str | Path, target: str | Path, max_shard_size: str = "100MB"):
    """
    Converts an xml file to an arrow file.
    :param source: The source directory containing the xml files.
    :param target: The target parquet file.
    :param max_shard_size: The maximum size of each shard. Must have a number followed by unit, e.g., "100 MB"
    """
    dataset = load_dataset("xml", data_dir=source)
    dataset.save_to_disk(target, max_shard_size=max_shard_size)


def get_data_path(dataset_name: str, data_type: str) -> str | Path:
    if data_type != "dataset" and data_type != "embeddings":
        raise ValueError("datatype must be 'dataset' or 'embeddings'")
    project_root = Path(__file__).resolve().parent.parent
    with open(project_root / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return project_root / cfg["paths"]["data"].format(dataset_name=dataset_name, data_type=data_type)

def load_local(dataset_name: str, data_type: str) -> Dataset | DatasetDict | Any:
    path = get_data_path(dataset_name, data_type)
    return load_from_disk(path) if data_type == "dataset" else torch.load(path / "embeddings.pt")

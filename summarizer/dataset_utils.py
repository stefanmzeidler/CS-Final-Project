import os
from datasets import load_dataset
from pathlib import Path
import json

def xml_to_arrow(source: str | os.PathLike[str], target: str | os.PathLike[str], max_shard_size:str = "100MB"):
    """
    Converts an xml file to an arrow file.
    :param source: The source directory containing the xml files.
    :param target: The target parquet file.
    :param max_shard_size: The maximum size of each shard. Must have a number followed by unit, e.g., "100 MB"
    """
    dataset = load_dataset("xml", data_dir = source)
    dataset.save_to_disk(target,max_shard_size = max_shard_size)

def get_data_path(datatype):
    project_root = Path(__file__).resolve().parent.parent
    with open(project_root / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return project_root / cfg["paths"]["data"][datatype]

def load_local(dataset):
    project_root = Path(__file__).resolve().parent.parent
    with open(project_root / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return project_root / cfg["paths"]["data"]["datasets"][dataset]
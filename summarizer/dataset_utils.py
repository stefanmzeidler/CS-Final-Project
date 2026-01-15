import os
from typing import Any
from datasets import load_from_disk, Dataset, DatasetDict
from pathlib import Path
import json
import torch
import lxml
from lxml import etree


def xml_to_arrow(source: str | Path, target: str | Path, max_shard_size = "100MB"):
    """
    Converts PMC xml files to a arrow files.
    :param source: The source directory containing the xml files.
    :param target: The target parquet directory for the arrow files.
    :param max_shard_size: The maximum size of the shards to create. Must be in "100MB" format.
    """
    def iter_pmc_articles(xml_dir):
        def check_text(text_list):
            return None if len(text_list) == 0 else " ".join(text_list).strip()

        for xml_file in Path(xml_dir).rglob("*.xml"):
            try:
                xml_file = str(xml_file)
                root = etree.parse(xml_file).getroot()
                pmcid = check_text(root.xpath("//article-id[@pub-id-type='pmc']//text()"))
                title = check_text(root.xpath("//title-group/article-title/text()"))
                abstract = check_text(root.xpath("//abstract/p/text()"))
                body_text = check_text(root.xpath("//body//text()"))
            except lxml.etree.XMLSyntaxError:
                continue
            if not pmcid or not title or not abstract or not body_text:
                continue
            yield {
                "pmcid": pmcid,
                "title": title,
                "abstract": abstract,
                "body_text": body_text,
            }
    ds = Dataset.from_generator(iter_pmc_articles, gen_kwargs = {"xml_dir": source})
    ds.save_to_disk(target,max_shard_size = max_shard_size)
    return ds

def get_data_path(dataset_name: str, data_type: str) -> str | Path:
    if data_type != "dataset" and data_type != "embeddings":
        raise ValueError("datatype must be 'dataset' or 'embeddings'")
    project_root = Path(__file__).resolve().parent.parent
    with open(project_root / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return project_root / cfg["paths"]["data"].format(
        dataset_name=dataset_name, data_type=data_type
    )


def load_local(dataset_name: str, data_type: str) -> Dataset | DatasetDict | Any:
    path = get_data_path(dataset_name, data_type)
    return (
        load_from_disk(path)
        if data_type == "dataset"
        else torch.load(path / "embeddings.pt")
    )


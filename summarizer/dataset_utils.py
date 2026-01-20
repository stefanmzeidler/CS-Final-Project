from datasets import load_from_disk, Dataset, DatasetDict
from pathlib import Path
import json
import torch
from Bio import Entrez
import os
from typing import Any, Dict,Iterator
import lxml
from lxml import etree
from dotenv import load_dotenv
from http.client import HTTPResponse
import re

def xml_to_arrow(source: str | Path, target: str | Path, max_shard_size="100MB") -> Dataset:
    """
    Converts PMC xml files to a arrow files.
    :param source: The source directory containing the xml files.
    :param target: The target parquet directory for the arrow files.
    :param max_shard_size: The maximum size of the shards to create. Must be in "100MB" format.
    """

    def iter_pmc_articles(xml_dir)->Iterator[Dict[str,str]]:
        for xml_file in Path(xml_dir).rglob("*.xml"):
            try:
                yield _parse_pmc_file(xml_file)
            except lxml.etree.XMLSyntaxError:
                continue
            except RuntimeError:
                continue

    ds = Dataset.from_generator(iter_pmc_articles, gen_kwargs={"xml_dir": source})
    ds.save_to_disk(target, max_shard_size=max_shard_size)
    return ds


def _parse_pmc_file(xml_file, xml_string=False) -> Dict[str, str]:
    """
    Parses an xml file or string for a PMC article and returns a dictionary containing the pmcid, abstract, title and body text.
    :param xml_file: The xml file to parse.
    :param xml_string: Whether file is in string format.
    :return: A dictionary containing the key-values pairs for pmcid, abstract, title and body_text.
    """
    def check_text(text_list):
        if len(text_list) > 0:
            cleaned_text = [clean_text(text) for text in text_list]
            return " ".join(cleaned_text).strip()
        return None

    if xml_string:
        root = etree.fromstring(xml_file)
    else:
        xml_file = str(xml_file)
        root = etree.parse(xml_file).getroot()
    pmcid = check_text(root.xpath("//article-id[1]/text()[1]"))
    title = check_text(root.xpath("//title-group[1]/article-title[1]/text()[1]"))
    abstract = check_text(root.xpath("//abstract//text()"))
    body_text = check_text(root.xpath("//body//text()"))
    if not pmcid or not title or not abstract or not body_text:
        raise RuntimeError("PMC XML parsing failed")
    return {
        "pmcid": pmcid,
        "title": title,
        "abstract": abstract,
        "body_text": body_text,
    }


def _project_root()->Path:
    return Path(__file__).resolve().parent.parent


def _open_config() -> Any:
    with open(_project_root() / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def get_data_path(dataset_name: str, data_type: str) -> str | Path:
    """
    Returns the path for the data directory for the specified dataset and type of data.
    :param dataset_name: The name of the dataset.
    :param data_type: Which data type to return. Either dataset or embeddings.
    :return: The path to the data directory.
    """
    if data_type != "dataset" and data_type != "embeddings":
        raise ValueError("datatype must be 'dataset' or 'embeddings'")
    cfg = _open_config()
    return _project_root() / cfg["paths"]["data"].format(
        dataset_name=dataset_name, data_type=data_type
    )


def get_audiences() -> Dict[str, str]:
    """
    Returns the audience types and LLM strings as a dictionary.
    :return: The audience types and LLM strings.
    """
    return _open_config()["audiences"]


def load_local(dataset_name: str, data_type: str) -> Dataset | DatasetDict | Any:
    """
    Loads a local dataset or the embeddings file for that dataset.
    :param dataset_name: The name of the dataset.
    :param data_type: Either "dataset" or "embeddings". Specifies which type.
    :return: The dataset as a dataset object or embeddings using torch.load functionality.
    """
    path = get_data_path(dataset_name, data_type)
    return (
        load_from_disk(path)
        if data_type == "dataset"
        else torch.load(path / "embeddings.pt")
    )


def _get_article(pmcid: str) -> str:
    """
    Given a PMC article ID, fetches the article using the Entrez API and returns an xml string for the article.
    :param pmcid: The PMC article ID.
    :return: The article XML string.
    """
    load_dotenv()
    Entrez.email = os.getenv("ENTREZ_EMAIL")
    handle = Entrez.efetch(db="pmc", id=pmcid)
    xml_string = HTTPResponse.read(handle).decode("utf-8")
    handle.close()
    return xml_string


def article_to_dict(pmcid: str)-> Dict[str, str]:
    """
    Takes an xml string for a PMC article and returns it as a dictionary containing the key-value pairs for pmcid, title, abstract, and body_text.
    :param pmcid: The PMC article ID.
    :return:
    """
    xml_string = _get_article(pmcid)
    article_dict = _parse_pmc_file(xml_string, xml_string=True)
    return article_dict


def clean_text(text) -> str:
    """
    Removes any extra whitespace from a string as well as any formatting tags.
    :param text: Text to clean.
    :return: Cleaned text.
    """
    pattern = r"[\r\n\t]|{.*?}|\\.*?{.*?}"
    return re.sub(pattern, " ", text)


def dict_to_string(doc, include_keys=False):
    """
    Takes an article dictionary and returns it as a string.
    :param doc: The article dictionary.
    :param include_keys: Whether to include the dictionary keys in the string.
    :return: The article string.
    """
    if include_keys:
        article_text = []
        for key, value in doc.items():
            article_text.append(f"{key}: {value}")
        return "\n".join(article_text)
    return " ".join(([value for value in doc.values()]))

from docretriever import DocRetriever
import json
import os
from datasets import load_dataset
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import torch


# Adapted from https://github.com/huggingface/sentence-transformers/blob/main/examples/sentence_transformer/applications/semantic-search/semantic_search_publications.py
class STRetriever(DocRetriever):
    def __init__(
        self,
        model="allenai-specter",
        dataset="uiyunkim-hub/pubmed-abstract",
        split="train",
    ):
        """
        Constructs a STRetriever instance. Use sbert transformer for creating embeddings and find similar documents.
        :param model: The name of the Sentence Transformers model to use. Default is allenai-specter.
        :param dataset: The HuggingFace dataset to use. Must be in namespace/dataset format. Default is uiyunkim-hub/pubmed-abstract.
        :param split: The dataset split to use. Default is train.
        """
        self.model = SentenceTransformer(model)
        print("Loading dataset")
        self.papers = load_dataset(dataset)[split]
        print("Dataset loaded")
        self.corpus_embeddings = None

    def retrieve_similar(self, query):
        # TODO
        ...

    def create_embeddings(self):
        print("Creating embeddings")
        paper_texts = [paper["abstract"] for paper in self.papers]
        self.corpus_embeddings = self.model.encode_document(
            paper_texts, convert_to_tensor=True
        )
        print("Embeddings created")
        print("Saving embeddings")
        embeddings_path = STRetriever.get_embeddings_path()
        with open(
            os.path.join(embeddings_path, "embeddings.pt"), "wb"
        ) as embeddings_file:
            torch.save(self.corpus_embeddings, embeddings_file)
        print("Embeddings saved")

    def load_embeddings(self, **kwargs):
        embeddings_path = STRetriever.get_embeddings_path()
        with open(
            os.path.join(embeddings_path, "embeddings.pt"), "wb"
        ) as embeddings_file:
            self.corpus_embeddings = torch.load(embeddings_file)

    @staticmethod
    def get_embeddings_path():
        project_root = Path(__file__).resolve().parent
        with open(project_root / "config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return project_root / cfg["paths"]["data"]["embeddings"]

    def search_papers(self, title, abstract):
        query_embedding = self.model.encode(
            title + "[SEP]" + abstract, convert_to_tensor=True
        )
        search_hits = util.semantic_search(query_embedding, self.corpus_embeddings)
        search_hits = search_hits[0]
        related_papers = []
        for hit in search_hits:
            related_papers.append(self.papers[hit["corpus_id"]])
        return related_papers


my_retriever = STRetriever(dataset="uiyunkim-hub/pubmed-abstract")
my_retriever.create_embeddings()

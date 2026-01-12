from docretriever import DocRetriever
import json
import os
from datasets import load_dataset,load_from_disk
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import torch


# Adapted from https://github.com/huggingface/sentence-transformers/blob/main/examples/sentence_transformer/applications/semantic-search/semantic_search_publications.py
# https://github.com/huggingface/sentence-transformers/blob/main/examples/sentence_transformer/applications/computing-embeddings/computing_embeddings_multi_gpu.py
class STRetriever(DocRetriever):
    def __init__(
        self,
        model="allenai-specter",
        dataset="uiyunkim-hub/pubmed-abstract",
        split="train",
        load_local=False
    ):
        """
        Constructs a STRetriever instance. Use sbert transformer for creating embeddings and find similar documents.
        :param model: The name of the Sentence Transformers model to use. Default is allenai-specter.
        :param dataset: The dataset to use. If from_disk is true, searches for an existing dataset file. Otherwise, should be the HuggingFace dataset to use. Must be in namespace/dataset format. Default is uiyunkim-hub/pubmed-abstract.
        :param split: The dataset split to use. Default is train.
        :param load_local: Whether to load the dataset from disk. Default is False.
        """
        print("Loading model")
        self.model = SentenceTransformer(model)
        print("Model loaded")
        print("Loading dataset")
        if load_local:
            self.papers = load_from_disk(STRetriever.get_data_path("datasets"))
        else:
            self.papers = load_dataset(dataset)[split]
            self.papers.save_to_disk(dataset_path = STRetriever.get_data_path("datasets"),max_shard_size="100MB")
        print("Dataset loaded")
        self.corpus_embeddings = None

    def retrieve_similar(self, query):
        # TODO
        ...

    def _yield_text(self):
        for paper in self.papers:
            yield paper["abstract"]

    def create_embeddings(self,max_papers):
        print("Creating embeddings")
        # paper_texts = [paper["abstract"] for paper in self.papers]
        paper_texts = []
        for index, text in enumerate(self._yield_text()):
            paper_texts.append(text)
            if index >= max_papers:
                break
        pool = self.model.start_multi_process_pool()
        embeddings_path = STRetriever.get_data_path("embeddings")
        self.corpus_embeddings = self.model.encode_document(sentences = paper_texts,pool=pool, normalize_embeddings=True, convert_to_tensor=True,show_progress_bar=True)
        torch.save(self.corpus_embeddings, os.path.join(embeddings_path, "embeddings.pt"))
        print("Embeddings saved")
        self.model.stop_multi_process_pool(pool)

    def _create_embeddings_shard(self, batch, shard_index,pool,embeddings_path):
        self.corpus_embeddings = self.model.encode_document(sentences = batch,pool=pool, normalize_embeddings=True, convert_to_tensor=True,show_progress_bar=True)
        torch.save(self.corpus_embeddings, os.path.join(embeddings_path, f"embeddings_{shard_index}.pt"))



    def load_embeddings(self, **kwargs):
        embeddings_path = STRetriever.get_data_path("embeddings")
        with open(
            os.path.join(embeddings_path, "embeddings.pt"), "wb"
        ) as embeddings_file:
            self.corpus_embeddings = torch.load(embeddings_file)



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

if __name__ == "__main__":
    my_retriever = STRetriever(dataset="uiyunkim-hub/pubmed-abstract", load_local= True)
    my_retriever.create_embeddings(100000)

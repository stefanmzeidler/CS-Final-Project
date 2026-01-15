from docretriever import DocRetriever
import os
from datasets import load_dataset, IterableDataset
from numpy import ndarray
from sentence_transformers import SentenceTransformer, util
import torch
from torch import Tensor
from typing import List, Generator, Any
import dataset_utils as du
from pathlib import Path


# Adapted from https://github.com/huggingface/sentence-transformers/blob/main/examples/sentence_transformer/applications/semantic-search/semantic_search_publications.py
# https://github.com/huggingface/sentence-transformers/blob/main/examples/sentence_transformer/applications/computing-embeddings/computing_embeddings_multi_gpu.py
class STRetriever(DocRetriever):
    def __init__(
            self,
            model="allenai-specter",
            dataset_name="uiyunkim-hub/pubmed-abstract",
            split="train",
            data_columns= None,
            load_local=False
    ):
        """
        Constructs a STRetriever instance. Use sbert transformer for creating embeddings and find similar documents.
        :param model: The name of the Sentence Transformers model to use. Default is allenai-specter.
        :param dataset_name: The name of the dataset to use. If load_local is true, searches for an existing dataset. Otherwise, should be the HuggingFace dataset to use. Must be in namespace/dataset format. Default is uiyunkim-hub/pubmed-abstract.
        :param split: The dataset split to use. Default is train.
        :param load_local: Whether to load the dataset from disk. Default is False.
        """
        if data_columns is None:
            self.data_columns = ["abstract"]
        else:
            self.data_columns = data_columns
        self.dataset_name = dataset_name
        print("Loading model")
        self.model = SentenceTransformer(model)
        print("Model loaded")
        print("Loading dataset")
        if load_local:
            self.papers = du.load_local(self.dataset_name, "dataset")
        else:
            self.papers = load_dataset(self.dataset_name, streaming = True)[split]
            self.papers.save_to_disk(dataset_path=du.get_data_path(self.dataset_name, "dataset"), max_shard_size="100MB")
        print("Dataset loaded")
        self.corpus_embeddings = None

    def retrieve_similar(self, query: str| List[str]) -> list[Any]:
        """
        Given a document, return a list of top 10 similar documents.
        :param query: The document query to search for.
        :return: List of related documents from the retriever dataset.
        """
        query_embeddings = self._get_embeddings(query)
        hits = util.semantic_search(query_embeddings, self.corpus_embeddings, score_function=util.dot_score)
        hits = hits[0]
        related_papers = []
        for hit in hits:
            related_papers.append(self.papers.select[hit["corpus_id"]])
        return related_papers

    def _yield_text(self) -> Generator[str | List[str], None, None]:
        for paper in self.papers:
            sections = []
            for col in self.data_columns:
                sections.append(paper[col])
            yield "[SEP]".join(sections)

    def create_corpus_embeddings(self, max_papers: int = None):
        print("Generating corpus embeddings")
        paper_texts = []
        if max_papers is None:
            max_papers = self.papers.shape[0]
        for index, text in enumerate(self._yield_text()):
            paper_texts.append(text)
            if index >= max_papers:
                break
        self.corpus_embeddings = self._get_embeddings(paper_texts)
        embeddings_path = du.get_data_path(self.dataset_name, "embeddings")
        embeddings_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.corpus_embeddings, os.path.join(embeddings_path, "embeddings.pt"))
        print("Corpus embeddings saved to disk")

    def _get_embeddings(self, paper_texts: str| List[str] | IterableDataset) -> List[Tensor]| ndarray| Tensor:
        print("Creating embeddings")
        pool = self.model.start_multi_process_pool()
        embeddings = self.model.encode_document(sentences=paper_texts, pool=pool, normalize_embeddings=True,
                                                convert_to_tensor=True, show_progress_bar=True)
        print("Embeddings created")
        self.model.stop_multi_process_pool(pool)
        return embeddings

    def load_embeddings(self):
        print("Loading embeddings")
        self.corpus_embeddings =  du.load_local(dataset_name = self.dataset_name, data_type="dataset")
        print("Embeddings loaded")

if __name__ == "__main__":
    my_retriever = STRetriever(dataset_name="PMC010xxxxxx", load_local=True)
    # my_retriever.create_corpus_embeddings()
    my_retriever.load_embeddings()

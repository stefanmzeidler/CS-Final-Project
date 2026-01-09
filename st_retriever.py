from docretriever import DocRetriever
import json
import os
from datasets import load_dataset

from sentence_transformers import SentenceTransformer, util


# Adapted from https://github.com/huggingface/sentence-transformers/blob/main/examples/sentence_transformer/applications/semantic-search/semantic_search_publications.py
class STRetriever(DocRetriever):
    def __init__(self, **kwargs):
        """
        Constructs a STRetriever object. This object uses a SentenceTransformer to create embeddings and retriever similar documents in the specified corpus.
        :param kwargs: dataset should be the HF repository namespace and dataset name, e.g., namespace/dataset
        split should be the dataset split to use. Defaults to train.
        model_name should be the name of the model to use. Will default to allenai-specter
        """
        if "dataset" not in kwargs:
            raise ValueError("dataset must be provided")
        if "split" in kwargs:
            split = kwargs["split"]
        else:
            split = "train"
        if "model_name" in kwargs:
            self.model = SentenceTransformer(kwargs["model_name"])
        else:
            self.model = SentenceTransformer("allenai-specter")
        print("Loading dataset")
        self.papers = load_dataset(kwargs["dataset"], streaming=True)[split]
        print("Dataset loaded")
        print("Creating embeddings")
        paper_texts = [paper["abstract"] for paper in self.papers]
        self.corpus_embeddings = self.model.encode(paper_texts, convert_to_tensor=True)
        print("Embeddings created")

    def retrieve_similar(self, query):
        # TODO
        ...

    # def load_dataset(self, type, dataset_files):
    #     match type:
    #         case "json":
    #             dataset_file = dataset_files[0]
    #             if self.model is None:
    #                 raise Exception("No model loaded")
    #             if not os.path.exists(dataset_file):
    #                 raise FileNotFoundError(dataset_file)
    #             with open(dataset_file) as fIn:
    #                 self.papers = json.load(fIn)
    #             if self.papers is None:
    #                 raise Exception("No papers in dataset")
    #             print(len(self.papers), "papers loaded")
    #             paper_texts = [paper["title"] + "[SEP]" + paper["abstract"] for paper in self.papers]
    #             self.corpus_embeddings = self.model.encode(paper_texts, convert_to_tensor=True)

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

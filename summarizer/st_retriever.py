import os
from datasets import load_dataset, IterableDataset
from numpy import ndarray
from sentence_transformers import SentenceTransformer, util
import torch
from torch import Tensor
from typing import List, Generator
import dataset_utils as du


# Adapted from https://github.com/huggingface/sentence-transformers/blob/main/examples/sentence_transformer/applications/semantic-search/semantic_search_publications.py
# https://github.com/huggingface/sentence-transformers/blob/main/examples/sentence_transformer/applications/computing-embeddings/computing_embeddings_multi_gpu.py
# https://github.com/huggingface/sentence-transformers/blob/main/examples/sentence_transformer/applications/retrieve_rerank/in_document_search_crossencoder.py
class STRetriever:
    def __init__(
        self,
        model_name="allenai-specter",
        dataset_name="PMC010xxxxxx",
        split="train",
        data_columns=None,
        load_local=True,
    ):
        """
        Constructs a STRetriever instance. Use sbert transformer for creating embeddings and find similar documents.
        :param model_name: The name of the Sentence Transformers model to use. Default is allenai-specter.
        :param dataset_name: The name of the dataset to use. If load_local is true, searches for an existing dataset. Otherwise, should be the HuggingFace dataset to use. Must be in namespace/dataset format. Default is uiyunkim-hub/pubmed-abstract.
        :param split: The dataset split to use. Default is train.
        :param data_columns: The column names to use for creating embeddings. Default is "title", "abstract", "body_text".
        :param load_local: Whether to load the dataset from disk. Default is False.
        """
        if data_columns is None:
            self.data_columns = ["title", "abstract", "body_text"]
        else:
            self.data_columns = data_columns
        self.dataset_name = dataset_name
        print("Loading retriever model")
        self.model = SentenceTransformer(model_name)
        print("Model loaded")
        print("Loading dataset")
        if load_local:
            self.papers = du.load_local(self.dataset_name, "dataset")
        else:
            self.papers = load_dataset(self.dataset_name)[split]
            self.papers.save_to_disk(
                dataset_path=du.get_data_path(self.dataset_name, "dataset"),
                max_shard_size="100MB",
            )
        print("Dataset loaded")
        self.corpus_embeddings = None

    def supporting_docs(self, query, pmcid: str, top_k: int) -> list[dict[str, str]]:
        """
        Given a PMC article, find the top_k similar articles from the dataset using bi-encoding.
        :param query: The article query.
        :param pmcid: The PMC ID for the article.
        :param top_k: The number of similar articles to return.
        :return: A list of dictionaries with the keys  'pmcid','title', 'abstract', 'body_text' for the similar articles.
        """
        print("Retrieving supporting documents")
        query_embeddings = self._get_embeddings(query)
        hits = util.semantic_search(
            query_embeddings,
            self.corpus_embeddings,
            score_function=util.dot_score,
            top_k=top_k,
        )
        hits = hits[0]
        supporting_docs = []
        for hit in hits:
            related_paper = self.papers[int(hit["corpus_id"])]
            if related_paper["pmcid"] == pmcid:
                continue
            supporting_docs.append(related_paper)
        print("Supporting docs retrieved")
        return supporting_docs

    def _yield_text(self, papers) -> Generator[str | List[str], None, None]:
        """
        Generates the text in an article joined by a [SEP] token for generating embeddings.
        :param papers: The papers to yield text for.
        """
        for paper in papers:
            sections = []
            for col in self.data_columns:
                sections.append(paper[col])
            yield "[SEP]".join(sections)

    def _create_paper_list(self, papers, max_papers: int = None) ->List[str]:
        """
        Creates list of articles as strings for creating embeddings.
        :param papers: The papers to generate embeddings for.
        :param max_papers: The max number of articles to process.
        :return: List of articles as strings.
        """
        paper_texts = []
        if max_papers is None:
            max_papers = papers.shape[0]
        for index, text in enumerate(self._yield_text(papers)):
            paper_texts.append(text)
            if index >= max_papers:
                break
        return paper_texts

    def create_corpus_embeddings(self, max_papers: int = None) -> Tensor:
        """
        Iterates through the PMC articles in the dataset and generates embeddings for each, saves the embeddings as a .pt file, and returns the embeddings as a Tensor file.
        :param max_papers: The maximum number of articles to process.
        :return: A Tensor object of the embeddings.
        """
        print("Generating corpus embeddings")
        paper_texts = self._create_paper_list(self.papers, max_papers)
        embeddings = self._get_embeddings(paper_texts)
        embeddings_path = du.get_data_path(self.dataset_name, "embeddings")
        embeddings_path.mkdir(parents=True, exist_ok=True)
        torch.save(embeddings, os.path.join(embeddings_path, "embeddings.pt"))
        print("Corpus embeddings saved to disk")
        return embeddings

    def _get_embeddings(self, paper_texts: str | List[str] | IterableDataset) -> Tensor:
        """
        Creates the embeddings for the given text.
        :param paper_texts: The text to create embeddings for.
        :return: A Tensor object of the embeddings.
        """
        print("Creating embeddings")
        pool = self.model.start_multi_process_pool()
        embeddings = self.model.encode_document(
            sentences=paper_texts,
            pool=pool,
            normalize_embeddings=True,
            convert_to_tensor=True,
            show_progress_bar=True,
        )
        print("Embeddings created")
        self.model.stop_multi_process_pool(pool)
        return embeddings

    def load_embeddings(self):
        """
        Loads the corpus embeddings for the dataset from a .pt file if it exists. Otherwise, creates and uses corpus embeddings via generate_corpus_embeddings using all the articles in the dataset.
        """
        embeddings_path = du.get_data_path(self.dataset_name, "embeddings")
        if not embeddings_path.exists() or not any(embeddings_path.iterdir()):
            print("No embeddings found.")
            self.corpus_embeddings = self.create_corpus_embeddings()
        else:
            print("Loading embeddings")
            self.corpus_embeddings = du.load_local(
                dataset_name=self.dataset_name, data_type="embeddings"
            )
            print("Embeddings loaded")

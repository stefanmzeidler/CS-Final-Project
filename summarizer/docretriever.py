from abc import ABC, abstractmethod
from typing import List, Generator, Any


class DocRetriever(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def retrieve_similar(self,query) -> List[Any]:
        """
        Given a document, return a list of top 10 similar documents.
        :param query: The document query to search for.
        :return: List of related documents from the retriever dataset.
        """
        pass

    @abstractmethod
    def create_corpus_embeddings(self, **kwargs):
        pass

    @abstractmethod
    def load_embeddings(self,**kwargs):
        pass

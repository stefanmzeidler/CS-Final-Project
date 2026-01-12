from abc import ABC, abstractmethod


class DocRetriever(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def retrieve_similar(self,query):
        pass

    @abstractmethod
    def create_corpus_embeddings(self, **kwargs):
        pass

    @abstractmethod
    def load_embeddings(self,**kwargs):
        pass

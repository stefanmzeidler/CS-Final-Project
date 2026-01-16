from abc import ABC, abstractmethod
import dataset_utils as du

class DocRetriever(ABC):
    @abstractmethod
    def __init__(self, model, dataset_name, split, data_columns, load_local):
        pass

    @abstractmethod
    def retrieve_similar(self, pmcid:str) -> list[dict[str, str]]:
        """
        Retrieves the top 10 similar documents from the dataset to the article specified by pmcid.
        :param pmcid: The pubmed central ID for the article.
        """
        pass

    @abstractmethod
    def create_corpus_embeddings(self, **kwargs):
        pass

    @abstractmethod
    def load_embeddings(self,**kwargs):
        pass

    @staticmethod
    def embeddings_check(self, dataset_name):
        embeddings_path = du.get_data_path(dataset_name, "embeddings")
        return embeddings_path.exists() and  any(embeddings_path.iterdir())


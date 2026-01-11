from abc import ABC, abstractmethod
from pathlib import Path
import json

class DocRetriever(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def retrieve_similar(self,query):
        pass

    @abstractmethod
    def create_embeddings(self, **kwargs):
        pass

    @abstractmethod
    def load_embeddings(self,**kwargs):
        pass

    @staticmethod
    def get_data_path(datatype):
        project_root = Path(__file__).resolve().parent
        with open(project_root / "config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return project_root / cfg["paths"]["data"][datatype]
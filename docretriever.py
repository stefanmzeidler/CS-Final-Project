from abc import ABC, abstractmethod

class DocRetriever(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def retrieve_similar(self,doc):
        pass

    @abstractmethod
    def build(self):
        pass

    

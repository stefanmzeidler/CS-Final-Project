from abc import ABC, abstractmethod

class LLM(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def prompt(self,doc,similar_docs):
        pass


from typing import List, Dict
import dataset_utils as du
from sentence_transformers.cross_encoder import CrossEncoder

class Reranker:
    def __init__(self,model_name = "cross-encoder/ms-marco-MiniLM-L12-v2"):
        """
        Class used to rerank a set of documents using specified model.
        :param model_name: The name of the model to use. Must be a SentenceTransformer cross-encoder model.
        """
        print("Loading reranker model")
        self.model = CrossEncoder(model_name)
        print("Model loaded")

    def rerank(self, query:str, supporting_docs:List[Dict[str, str]], top_k:int) ->List[Dict[str, str]]:
        """
        Given a query article, ranks the top_k most similar documents from a list of supporting documents.
        :param query: The article query.
        :param supporting_docs: The documents to rerank.
        :param top_k: The number of documents to rerank.
        :return: Subset of supporting_docs.
        """
        print("Re-ranking documents")
        model_inputs = [[query, du.dict_to_string(doc)] for doc in supporting_docs]
        scores = self.model.predict(model_inputs,show_progress_bar=True)
        results = [
            {"doc": doc, "score": score} for doc, score in zip(supporting_docs, scores)
        ]
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        print("Documents re-ranked")
        return [result["doc"] for result in results[:min(top_k,len(results))]]
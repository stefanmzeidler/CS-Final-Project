from typing import List, Dict
import dataset_utils as du
from sentence_transformers.cross_encoder import CrossEncoder

class Reranker:
    def __init__(self,model_name = "cross-encoder/ms-marco-MiniLM-L12-v2"):
        print("Loading reranker model")
        self.model = CrossEncoder(model_name)
        print("Model loaded")

    def rerank(self, query:str, supporting_docs:List[Dict[str, str]], top_k:int) ->list[dict[str, str]]:
        print("Re-ranking documents")
        model_inputs = [[query,du.doc_to_string(doc)] for doc in supporting_docs]
        scores = self.model.predict(model_inputs,show_progress_bar=True)
        results = [
            {"doc": doc, "score": score} for doc, score in zip(supporting_docs, scores)
        ]
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        print("Documents re-ranked")
        return [result["doc"] for result in results[:min(top_k,len(results))]]
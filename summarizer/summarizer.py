from st_retriever import STRetriever
from reranker import Reranker
from llm import LLM
from typing import Dict
import dataset_utils as du


class Summarizer:
    def __init__(self, st_retriever: STRetriever, reranker: Reranker, llm: LLM):
        """
        Class that summarizes PubMedCentral articles for different audiences specified in the config file using an LLM and RAG.
        :param st_retriever: STRetriever used to retrieve articles.
        :param reranker: Reranker used to rerank retrieved articles..
        :param llm: The LLM used to summarize the article.
        """
        self.st_retriever = st_retriever
        self.st_retriever.load_embeddings()
        self.reranker = reranker
        self.llm = llm

    def summarize(
        self,
        pmcid: str,
        retrieval_top_k: int = 10,
        rerank_top_k: int = 3,
        max_words=300,
        temperature=0.8,
    )->Dict[str,str]:
        """
        Summarizes an article specified by PMC ID for different audiences using an LLM.
        :param pmcid: The PMC ID of the article.
        :param retrieval_top_k: The number of similar documents to retrieve. Must be greater or equal to rerank_top_k and positive.
        :param rerank_top_k: The number of similar documents to rerank and return. Must be less than or equal to retrieval_top_k and positive.
        :param max_words: The maximum number of words in the summaries.
        :param temperature: The temperature of the LLM.
        :return: The summaries for each audience as key-value pairs.
        """
        if retrieval_top_k < 1 or rerank_top_k < 1:
            raise ValueError("retrieval_top_k and rerank_top_k must be at least 1")
        paper = du.article_to_dict(pmcid)
        supporting_docs = []
        for doc in self._retrieve_and_rerank(
            paper, pmcid, rerank_top_k, retrieval_top_k
        ):
            supporting_docs.append(du.dict_to_string(doc, include_keys=True))
        supporting_docs = "\n".join(supporting_docs)
        llm_query = du.dict_to_string(paper, include_keys=True)
        summaries = self.llm.query_model(
            query=llm_query,
            supporting_docs=supporting_docs,
            audiences=du.get_audiences(),
            max_words=max_words,
            temperature=temperature,
        )
        return summaries

    def _retrieve_and_rerank(
        self,
        paper: dict[str, str | None],
        pmcid: str,
        rerank_top_k: int,
        retrieval_top_k: int,
    ) -> list[dict[str, str]]:
        search_query = du.dict_to_string(paper)
        supporting_docs_list = self.st_retriever.supporting_docs(
            query=search_query, pmcid=pmcid, top_k=retrieval_top_k
        )
        supporting_docs_list = self.reranker.rerank(
            query=search_query, supporting_docs=supporting_docs_list, top_k=rerank_top_k
        )
        return supporting_docs_list

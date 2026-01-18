from st_retriever import STRetriever
from reranker import Reranker
from llm import LLM
import dataset_utils as du


class Summarizer:
    def __init__(self, st_retriever: STRetriever, reranker: Reranker, llm: LLM):
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
    ):
        if retrieval_top_k < 1 or rerank_top_k < 1:
            raise ValueError("retrieval_top_k and rerank_top_k must be at least 1")
        paper = du.article_to_dict(pmcid)
        supporting_docs = []
        for doc in self.retrieve_and_rerank(
            paper, pmcid, rerank_top_k, retrieval_top_k
        ):
            supporting_docs.append(du.doc_to_string(doc, include_keys=True))
        supporting_docs = "\n".join(supporting_docs)
        llm_query = du.doc_to_string(paper, include_keys=True)
        summaries = self.llm.query_model(
            query=llm_query,
            supporting_docs=supporting_docs,
            audiences=du.get_audiences(),
            max_words=max_words,
            temperature=temperature,
        )
        return summaries

    def retrieve_and_rerank(
        self,
        paper: dict[str, str | None],
        pmcid: str,
        rerank_top_k: int,
        retrieval_top_k: int,
    ) -> list[dict[str, str]]:
        search_query = du.doc_to_string(paper)
        supporting_docs_list = self.st_retriever.supporting_docs(
            query=search_query, pmcid=pmcid, top_k=retrieval_top_k
        )
        supporting_docs_list = self.reranker.rerank(
            query=search_query, supporting_docs=supporting_docs_list, top_k=rerank_top_k
        )
        return supporting_docs_list


if __name__ == "__main__":
    from st_retriever import STRetriever

    my_summarizer = Summarizer(
        STRetriever(dataset_name="PMC010xxxxxx"), Reranker(), LLM()
    )
    summaries = my_summarizer.summarize("PMC10000014")
    print(summaries)
    with open("summaries.txt", "a") as f:
        for value in summaries.values():
            f.write(value + "\n")
    print(summaries['college'])

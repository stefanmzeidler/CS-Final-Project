from docretriever import DocRetriever
from llm import LLM
import dataset_utils as du

class Summarizer:
    def __init__(self, doc_retriever: DocRetriever):
        self.doc_retriever = doc_retriever
        self.doc_retriever.load_embeddings()
        self.llm = LLM()


    def summarize(self, pmcid):
        paper = du.article_to_dict(pmcid)
        supporting_docs = self.doc_retriever.supporting_docs(pmcid)
        print(Summarizer._concat_docs(supporting_docs))
        # self.llm.query_model(paper = paper, supporting_docs = supporting_docs, temperature=.8)
    @staticmethod
    def _concat_docs( supporting_docs):
        docs = []
        for index, doc in enumerate(supporting_docs):
            docs.append(f"Document {index}: {du.clean_text(doc['title'])}")
            docs.append(f"Abstract: {du.clean_text(doc['abstract'])}")
            docs.append(f"Body text: {du.clean_text(doc['body_text'])}")
        return "\n".join(docs)

if __name__ == '__main__':
    from st_retriever import STRetriever
    my_summarizer = Summarizer(STRetriever(dataset_name="PMC010xxxxxx"))
    my_summarizer.summarize("PMC10000014")

from docretriever import DocRetriever

class Summarizer:
    def __init__(self, doc_retriever: DocRetriever):
        self.doc_retriever = doc_retriever
        self.doc_retriever.load_embeddings()


    def summarize(self, pmcid):
        similar_docs = self.doc_retriever.retrieve_similar(pmcid)
        print(similar_docs[0]['pmcid'])

if __name__ == '__main__':
    from st_retriever import STRetriever
    my_summarizer = Summarizer(STRetriever(dataset_name="PMC010xxxxxx"))
    my_summarizer.summarize("PMC10000014")
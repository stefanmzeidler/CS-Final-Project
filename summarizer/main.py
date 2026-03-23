from summarizer import Summarizer
from st_retriever import STRetriever
from reranker import Reranker
from llm import LLM
import sys

if __name__ == "__main__":
    my_summarizer = Summarizer(STRetriever(),Reranker(),LLM())
    summaries = my_summarizer.summarize(sys.argv[1])
    for audience,summary in summaries.items():
        print(f"\033[31mFor {audience} audience:\033[0m")
        print(summary+"\n")



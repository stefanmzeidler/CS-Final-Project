# myapp/summarizer_service.py
from .Summarizer.summarizer import Summarizer

class SummarizerService:
    def __init__(self):
        print("Loading summarizer model")
        self.summarizer = Summarizer()
        print("Summarizer model loaded")

    def summarize(self, pmcid):
        print("Generating summaries")
        summaries = self.summarizer.summarize(pmcid)
        print("Summaries generated")
        return summaries

_service = None

def get_summarizer():
    global _service
    if _service is None:
        _service = SummarizerService()
    return _service
from google import genai
import os
from dotenv import load_dotenv

class LLM:
    def __init__(self, **kwargs):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)

    def prompt(self,doc,similar_docs):
        ...
# if __name__ == "__main__":
    # load_dotenv()
    # api_key = os.getenv("GEMINI_API_KEY")
    # client = genai.Client(api_key = api_key)
    # response = client.models.generate_content(model="gemini-3-flash-preview", contents="Explain how AI works in a few words")
    # print(response.text)

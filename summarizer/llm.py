from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
import yaml

class LLM:
    def __init__(self, **kwargs):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)

    def query_model(self, paper, supporting_docs, temperature):
        with open("prompts.yaml") as f:
            prompts = yaml.safe_load(f)
        query = self.generate_query(prompts = prompts, name ="summarize_paper", audience ="lay person with an 8th grade reading level", max_words = 300, paper = paper, supporting_docs = supporting_docs)
        return query
        # response = self.client.models.generate_content(
        #     model="gemini-2.0-flash",
        #     contents=query["text"],
        #     config= types.GenerateContentConfig(system_instruction=query["system"],temperature=temperature)
        # )
        # return response

    @staticmethod
    def generate_query(prompts, name, **kwargs):
        p = prompts["prompts"][name]
        system = p["system"]
        template = p["template"]
        return {"system": system.format(**kwargs), "text": template.format(**kwargs)}


if __name__ == "__main__":
    my_llm = LLM()
    my_llm.query_model("this is the doc", "this is supporting doc", temperature=0.8)

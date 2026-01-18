from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
import yaml
from collections import defaultdict
from typing import Dict


class LLM:
    def __init__(self, model_name="gemini-2.0-flash"):
        print("Initializing LLM")
        self.model_name = model_name
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
        print("Initialization done")

    def query_model(
        self,
        query: str,
        supporting_docs: str,
        audiences: Dict[str, str],
        max_words: int,
        temperature: float,
    ) -> Dict[str, str]:
        print("Querying llm")
        summaries = defaultdict(str)
        with open("prompts.yaml") as f:
            prompts = yaml.safe_load(f)

        for key, audience in audiences.items():
            query = self._generate_query(
                prompts=prompts,
                name="summarize_paper",
                audience=audience,
                max_words=max_words,
                paper=query,
                supporting_docs=supporting_docs,
            )
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=query["text"],
                config=types.GenerateContentConfig(
                    system_instruction=query["system"], temperature=temperature
                ),
            )
            summaries[key] = response.text
        print("Query done")
        return summaries

    @staticmethod
    def _generate_query(prompts, name, **kwargs):
        p = prompts["prompts"][name]
        system = p["system"]
        template = p["template"]
        return {"system": system.format(**kwargs), "text": template.format(**kwargs)}

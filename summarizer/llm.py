from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
import yaml
from collections import defaultdict
from typing import Dict


class LLM:
    def __init__(self, model_name="gemini-2.0-flash"):
        """
        Wrapper class for a gemini LLM.
        :param model_name: The name of the model.
        """
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
        """
        Asks the model to summarize the article query using the supporting_docs for  different audiences.
        :param query: The article query.
        :param supporting_docs: The supporting documents.
        :param audiences: The audiences
        :param max_words: The max number of words in the summary.
        :param temperature: The LLM temperature.
        :return: The summaries as key-value pairs for each audience.
        """
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
        """
        Loads the specified prompt template from the prompts, inserts the kwargs, and returns a dictionary.
        :param prompts: The prompts template as a python object.
        :param name: The name of the prompt.
        :param kwargs: The corresponding key-value pairs for the variables in the prompt.
        :return: A dictionary of formatting strings for the system and text values for the LLM query.
        """
        p = prompts["prompts"][name]
        system = p["system"]
        template = p["template"]
        return {"system": system.format(**kwargs), "text": template.format(**kwargs)}

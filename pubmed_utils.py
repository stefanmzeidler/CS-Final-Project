from Bio import Entrez
import json
from pathlib import Path
import xmltodict
from torchgen.api.cpp import return_type
import os
from dotenv import load_dotenv

def get_article(pmid):
    load_dotenv()

    Entrez.email = os.getenv("ENTREZ_EMAIL")
    handle = Entrez.efetch(db="pubmed",return_type ="medline", id=pmid)
    record = Entrez.read(handle)
    handle.close()
    return record

def pmid_to_pmcid(pmid: str) -> str | None:
    handle = Entrez.elink(
        dbfrom="pubmed",
        db="pmc",
        id=pmid
    )
    record = Entrez.read(handle)
    handle.close()

    try:
        return record[0]["LinkSetDb"][0]["Link"][0]["Id"]
    except (IndexError, KeyError):
        return None


article = get_article("19304878")
print(article)



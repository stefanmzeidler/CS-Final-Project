from Bio import Entrez
import os

from Bio.Entrez.Parser import DictionaryElement
from dotenv import load_dotenv

def get_article(pmid:str) -> DictionaryElement:
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
print(article["PubmedArticle"][0]["MedlineCitation"]["Article"]["Abstract"]["AbstractText"][0])
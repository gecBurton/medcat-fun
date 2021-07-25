from typing import List, Dict, Any

from medcat.vocab import Vocab
from medcat.cdb import CDB
from medcat.cat import CAT

# Load the vocab model you downloaded
vocab = Vocab.load("vocab.dat")
# Load the cdb model you downloaded
cdb = CDB.load("cdb-medmen-v1.dat")

# Create cat - each cdb comes with a config that was used
#to train it. You can change that config in any way you want, before or after creating cat.
cat = CAT(cdb=cdb, config=cdb.config, vocab=vocab)

# Test it
text = "My simple document with leukocytes in my kidney failure"
#text = 'i have bad leukocytes. check my white blood cells'
doc_spacy = cat(text)
# Print detected entities
print(doc_spacy.ents)

# Or to get an array of entities, this will return much more information
# and usually easier to use unless you know a lot about spaCy

from pydantic import BaseModel

class Entity(BaseModel):
    pretty_name: str
    cui: str
    tuis: List[str]
    types: List[str]
    source_value: str
    detected_name: str
    acc: float
    context_similarity: float
    start: int
    end: int
    icd10: List
    ontologies: List
    snomed: List
    id: int
    meta_anns: Any


class Response(BaseModel):
    entities: Dict[int, Entity]
    tokens: List


from fastapi import FastAPI
app = FastAPI()


@app.post("/parse-text/", response_model=Response)
def parse_text(txt: str):
    return cat.get_entities(txt)

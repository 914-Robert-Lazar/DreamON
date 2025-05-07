import spacy
from keyword_extraction.keyword_extraction_utils import load_nrc_lexicon,extract_emotions

emotion_lexicon = load_nrc_lexicon()
model_spacy = spacy.load("en_core_web_sm")

def keyword_extraction_spacy(text,model):
    doc = model(text)
    return {
        "Symbol": [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "LOC", "ORG", "NORP", "FAC","EVENT", "WORK_OF_ART"]],
        "Character": [ent.text for ent in doc.ents if ent.label_ == "PERSON"],
        "Emotion": extract_emotions(text,emotion_lexicon,model),
        "Action": [token.lemma_ for token in doc if token.pos_ == "VERB"],
        "Setting": [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC","FAC"]]
    }
from gliner import GLiNER

model_gliner = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

def keyword_extraction_gliner(text,model):
    dream_labels = ["Symbol", "Emotion", "Character", "Setting", "Action"]
    entities = model.predict_entities(text, dream_labels, threshold=0.5)
    output = {label: [] for label in dream_labels}
    for item in entities:
        entity_type = item['label']
        entity_text = item['text']
        output[entity_type].append(entity_text)
    return output

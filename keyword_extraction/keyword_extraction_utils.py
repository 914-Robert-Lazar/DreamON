from collections import defaultdict

def load_nrc_lexicon(path='NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'):
    emotion_map = defaultdict(set)
    with open(path, 'r') as f:
        for line in f:
            word, emotion, score = line.strip().split('\t')
            if score == '1':
                emotion_map[emotion].add(word)
    return emotion_map

def extract_emotions(text, lexicon):
    doc = nlp(text.lower())
    emotion_scores = defaultdict(int)
    for token in doc:
        for emotion, words in lexicon.items():
            if token.lemma_ in words:
                emotion_scores[emotion] += 1
    sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
    return [e[0] for e in sorted_emotions[:3]] if sorted_emotions else []
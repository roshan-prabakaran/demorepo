# nlp/train_chatbot.py
import json, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
from sklearn.preprocessing import LabelEncoder

MODEL_DIR = '../models' if os.path.exists('../models') else 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

INTENTS_PATH = 'intents.json'
with open(INTENTS_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

patterns = []
tags = []
for intent in data['intents']:
    for p in intent['patterns']:
        patterns.append(p)
        tags.append(intent['tag'])

# Encode tags
le = LabelEncoder()
y = le.fit_transform(tags)

# Pipeline: TF-IDF + Logistic Regression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2))),
    ('clf', LogisticRegression(max_iter=1000))
])

pipeline.fit(patterns, y)

# Save pipeline and label encoder
joblib.dump(pipeline, os.path.join(MODEL_DIR, 'chatbot_pipeline.joblib'))
joblib.dump(le, os.path.join(MODEL_DIR, 'chatbot_label_encoder.joblib'))

print("Saved chatbot pipeline and label encoder to models/")

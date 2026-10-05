"""
Inference module for Resume Classification
Loads the Dual TF-IDF + LinearSVC model from saved_model/
"""
import os
import re
import json
import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_model")

STOP_WORDS = set(ENGLISH_STOP_WORDS)

def preprocess(txt):
    """Clean and tokenize resume text."""
    txt = str(txt).lower()
    txt = re.sub(r"http\S+|www\.\S+", " ", txt)
    txt = re.sub(r"\S+@\S+", " ", txt)
    txt = re.sub(r"[^a-z\s]", " ", txt)
    tokens = [w for w in txt.split() if w not in STOP_WORDS and len(w) > 1]
    return " ".join(tokens)

def head_of(text, n=8):
    """Extract initial tokens (headline / job title)."""
    return " ".join(text.split()[:n])

class ResumePredictor:
    def __init__(self, model_dir=MODEL_DIR):
        self.clf = joblib.load(os.path.join(model_dir, "sk_classifier.joblib"))
        self.cfg = joblib.load(os.path.join(model_dir, "vectorizers.joblib"))
        with open(os.path.join(model_dir, "labels.json"), "r") as f:
            self.labels = json.load(f)
            
    def predict(self, raw_text, top_k=3):
        """
        Predict category for a raw resume string.
        Returns top-k predicted categories with confidence percentages.
        """
        clean = preprocess(raw_text)
        if not clean.strip():
            return []
            
        Xb = self.cfg["body"].transform([clean])
        Xh = self.cfg["head"].transform([head_of(clean, self.cfg.get("head_words", 8))])
        w = self.cfg.get("w", 1.0)
        X = hstack([Xb, Xh * w]).tocsr()
        
        scores = self.clf.decision_function(X)[0]
        
        # Softmax calibration for readable confidence percentages
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / np.sum(exp_scores)
        
        top_indices = np.argsort(-scores)[:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "category": self.labels[idx],
                "score": round(float(scores[idx]), 3),
                "confidence": round(float(probs[idx]) * 100, 1)
            })
            
        return results

# Singleton instance
_predictor = None

def predict_resume(raw_text, top_k=3):
    global _predictor
    if _predictor is None:
        _predictor = ResumePredictor()
    return _predictor.predict(raw_text, top_k=top_k)

if __name__ == "__main__":
    sample = """
    Full Stack Developer with 4 years experience in Python, Django, React, PostgreSQL, Docker, AWS.
    Built microservices and scalable web applications.
    """
    res = predict_resume(sample, top_k=3)
    print("Predictions for sample resume:")
    for r in res:
        print(f"  {r['category']}: {r['confidence']}% (score: {r['score']})")

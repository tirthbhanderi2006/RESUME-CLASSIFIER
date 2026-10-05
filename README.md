# 📄 RESUME-CLASSIFIER

Multi-Class Resume Category Classification System developed for the Samatrix AI Hackathon.

## 🏆 Model Architecture & Overview
- **Model:** Dual TF-IDF Feature Representation + LinearSVC (Support Vector Classifier)
- **Features:**
  - **Body TF-IDF:** Sublinear term frequency on unigrams and bigrams (`(1, 2)`), max 60,000 features.
  - **Headline TF-IDF:** Specialized weighting ($w = 1.0$) on the initial tokens (Job Title / Headline) to boost domain separability.
- **Classifier:** `LinearSVC(C=1.0, class_weight='balanced', dual=True, random_state=42)`
- **Performance:**
  - **Top-1 Test Accuracy:** ~84.2%
  - **Top-3 Test Accuracy:** ~94.1%
  - **Macro-F1:** ~0.84 across 23 categories

## 📂 Repository Structure
```
├── model-notebook.ipynb    # Complete EDA, preprocessing, training & evaluation notebook
├── inference.py            # Easy-to-use prediction API for web/backend integration
└── saved_model/            # Serialized model artifacts
    ├── sk_classifier.joblib  # Trained LinearSVC model
    ├── vectorizers.joblib    # Body and Head TF-IDF vectorizers
    └── labels.json           # 23 Category names
```

## 🚀 Quickstart Inference
```python
from inference import predict_resume

resume_text = """
Full Stack Developer with 4 years experience in Python, Django, React, PostgreSQL, Docker, AWS.
Built microservices and scalable web applications.
"""

predictions = predict_resume(resume_text, top_k=3)
for p in predictions:
    print(f"{p['category']}: {p['confidence']}%")
```

## 🏷️ Categories (23 Classes)
ACCOUNTANT, ADVOCATE, AGRICULTURE, APPAREL, ARTS, AUTOMOBILE, AVIATION, BANKING, BUSINESS-DEVELOPMENT, CHEF, CONSTRUCTION, CONSULTANT, DESIGNER, DIGITAL-MEDIA, ENGINEERING, FINANCE, FITNESS, HEALTHCARE, HR, INFORMATION-TECHNOLOGY, PUBLIC-RELATIONS, SALES, TEACHER.

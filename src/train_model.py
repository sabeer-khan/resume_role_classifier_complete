import os
from preprocess import preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load resumes
base_path = "../resume_data"
texts, labels = [], []
for label in os.listdir(base_path):
    for file in os.listdir(os.path.join(base_path, label)):
        with open(os.path.join(base_path, label, file), 'r', encoding='utf-8') as f:
            texts.append(f.read())
            labels.append(label)

# Preprocess
cleaned = [preprocess(t) for t in texts]

# TF-IDF
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(cleaned)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, stratify=labels)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

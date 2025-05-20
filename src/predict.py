import pickle
from preprocess import preprocess
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model and vectorizer
import joblib
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Predict function
def predict_resume(text):
    cleaned = preprocess(text)
    vec = vectorizer.transform([cleaned])
    return model.predict(vec)[0]

# Load new resume
with open("../new_resume.txt", "r") as f:
    resume = f.read()

print("Predicted Role:", predict_resume(resume))

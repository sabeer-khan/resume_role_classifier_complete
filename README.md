# 💼 Resume Role Classifier: How I Built an NLP Pipeline to Categorize Resumes

A complete NLP + Machine Learning project to automatically classify resumes into job roles using Python, spaCy, and scikit-learn.

---

## 🧠 Project Summary

Recruiters often deal with hundreds of resumes for each job post. Manually sorting them by role is time-consuming and error-prone. This project automates that process by using Natural Language Processing to classify resumes into roles like:

- Data Scientist  
- Frontend Developer  
- DevOps Engineer

It simulates a real-world ML pipeline — from raw text input to prediction — and is a strong addition to any ML portfolio.

---

## 📁 Folder Structure

resume_role_classifier/
├── resume_data/ # Sample .txt resumes by role
│ ├── data_scientist/
│ ├── frontend_developer/
│ └── devops_engineer/
├── src/ # Python scripts
│ ├── preprocess.py
│ ├── train_model.py
│ └── predict.py
├── new_resume.txt # Test input file
├── requirements.txt # Project dependencies
└── README.md # This documentation


---

## 🚀 How to Run the Project

### 1. 📦 Install Dependencies

pip install -r requirements.txt
python -m spacy download en_core_web_sm

## 2. 🧹 Preprocess + Train Model

cd src
python train_model.py

3. 🧪 Predict Role for a New Resume

Edit new_resume.txt with any resume text, then:

python predict.py

You’ll get output like:

Predicted Role: frontend_developer

📊 Model Overview

### 🧠 Project Pipeline

- 🔹 **Preprocessing with spaCy**  
  Tokenization, lemmatization, stopword & punctuation removal

- 🔹 **Feature extraction using TfidfVectorizer**  
  Converts text into meaningful numerical vectors

- 🔹 **Multiclass classification using LogisticRegression**  
  Simple, effective classifier for resume role prediction

- 🔹 **Evaluation via classification_report from scikit-learn**  
  Outputs precision, recall, F1-score, and accuracy


Example results:

### 📊 Classification Report

| Class              | Precision | Recall | F1-Score |
|--------------------|-----------|--------|----------|
| Data Scientist     | 1.00      | 0.83   | 0.91     |
| DevOps Engineer    | 0.75      | 1.00   | 0.86     |
| Frontend Developer | 1.00      | 1.00   | 1.00     |

**Overall Accuracy:** `91%`


📄 Sample Resume File

Each .txt resume is stored in a labeled folder:

resume_data/
├── data_scientist/
│   ├── ds1.txt
│   └── ds2.txt

The folder name serves as the classification label.

🧠 What I Learned

✅ Building and structuring a real-world NLP classification pipeline
✅ Feature extraction from unstructured text
✅ Creating reproducible ML projects with clear documentation
✅ Preparing GitHub + Medium-ready content for employers and peers

🛠️ Tech Stack

### 🛠️ Tech Stack

- 🐍 **Python 3.10+**
- 🧠 **spaCy** – for text preprocessing (tokenization, lemmatization, etc.)
- 📊 **scikit-learn** – for vectorization, training, and evaluation
- 🧮 **TF-IDF** – to convert text into feature vectors
- 🎯 **Logistic Regression** – used as the classification model


📖 Medium Article

Want a full walkthrough?
📚 Read the article on Medium
https://medium.com/@sabeerkhan1603/resume-role-classifier-how-i-built-an-nlp-pipeline-to-categorize-resumes-f7b5b0a44bcd












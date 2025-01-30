import os
import re
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk import download
import time

# Download NLTK stopwords
download('stopwords')
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    unwanted_words = {'islam', 'father', 'religion', 'marital status', 'blood group', 'date of birth', 'cnic', 'high school', 'matriculation', 'muzaffargarh', 'cgpa ', 'page', 'email', 'number', 'name', 'address', 'add', 'mobile', 'karachi', 'pakistan', 'gulshane', 'iqbal', 'shah faisal town', 'quetta'}
    text = text.replace('/', ' ')
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([word for word in text.split() if word not in stop_words and word not in unwanted_words])
    return text

# Helper function to read files
def read_file(file):
    return file.read().decode('latin-1')

# Initialize labeled data
label_data = {
    "Umair Ahmed - CV for Lab Instructor.txt": 0,
    "Hafiz Ali Raja - CV for Lab Instructor.txt": 0,
    "Faisal Shahzad - CV for Lab Instructor.txt": 0,
    "Saad Hassan Khan - CV for Research Assistant (1).txt":1,
    "Waqas Ahmed - CV for Lab Instructor.txt":0,
    "Muhammad Omaid Sheikh - CV for for Lab Instructor.txt": 0,
    "Awais Anwar - CV for Lab Instructor.txt": 0,
    "Urooj Sheikh - CV for Lab Engineer.txt": 0,
    "Ghulam Jaffar - CV for Research Assistant.txt": 1,
    "Ebad Ali - CV for Research Assistant.txt": 1,
    "Sana Fatima - CV for Research Assistant.txt": 1,
    "Muhammad Tayyab Yaqoob - CV for Lab Engineer.txt": 0,
    "Haris Ahmed - CV for Research Assistant.txt": 1,
    "Faisal Nisar - CV for for Research Assistant.txt": 1,
    "Muhammad Azmi Umer - CV for Lab Instructor.txt":0,
    "Shawana Khan - CV for Lab Instructor (1).txt":0
}

# Title
st.title("Resume Classification App")
st.markdown("Upload resumes to classify them as either 'Lab Instructor' or 'Research Assistant'.")

# Sidebar for file uploads
st.sidebar.title("Upload Resumes")
uploaded_files = st.sidebar.file_uploader("Choose files", accept_multiple_files=True)

# Process uploaded labeled resumes
labeled_resumes = []
labels = []

# Handling labeled resumes
for filename, label in label_data.items():
    file = next((f for f in uploaded_files if f.name == filename), None)
    if file is not None:
        file_content = read_file(file)
        preprocessed_content = preprocess(file_content)
        labeled_resumes.append(preprocessed_content)
        labels.append(label)

if labeled_resumes:
    # Display loading animation
    with st.spinner('Processing and training the model...'):
        time.sleep(2)

    # Job descriptions
    jd_ra_file = next((f for f in uploaded_files if f.name == "JD Research Assistants.txt"), None)
    jd_li_file = next((f for f in uploaded_files if f.name == "JD-Instructors.txt"), None)

    if jd_ra_file and jd_li_file:
        job_description_ra = preprocess(read_file(jd_ra_file))
        job_description_li = preprocess(read_file(jd_li_file))

        # Vectorize the text using TF-IDF
        vectorizer = TfidfVectorizer()
        X_labeled = vectorizer.fit_transform(labeled_resumes)

        # Train a Naive Bayes classifier
        X_train, X_test, y_train, y_test = train_test_split(X_labeled, labels, test_size=0.2, random_state=48)
        clf = MultinomialNB()
        clf.fit(X_train, y_train)

        # Predict on the test set
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.success(f"Model trained with accuracy: {accuracy * 100:.2f}%")

        # Process and predict for new resumes (unlabeled data)
        if st.sidebar.button("Classify New Resumes"):
            unlabeled_files = [f for f in uploaded_files if f.name not in label_data and f.name not in ["JD Research Assistants.txt", "JD-Instructors.txt"]]
            unlabeled_resumes = [preprocess(read_file(file)) for file in unlabeled_files]
            X_unlabeled = vectorizer.transform(unlabeled_resumes)
            unlabeled_preds = clf.predict(X_unlabeled)

            st.markdown("### Prediction Results")
            for file, prediction in zip(unlabeled_files, unlabeled_preds):
                label = "Research Assistant" if prediction == 1 else "Lab Instructor"
                st.write(f"Resume: {file.name} -> Predicted label: {label}")

    else:
        st.error("Please upload both job description files.")

else:
    st.warning("Please upload labeled resumes to train the model.")

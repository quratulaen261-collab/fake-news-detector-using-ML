import pandas as pd
import numpy as np
import re
import streamlit as st
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("E:\\fake news detector using ML\\data.csv")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

df['text'] = df['text'].apply(clean_text)

X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

st.title("Fake News Detector")
st.write("Enter a news headline or short paragraph to check if it's fake or real:")

news = st.text_area("Enter the news text here:")

if st.button("Check"):
    if news.strip() == "":
        st.warning("Please enter some text first.")
    else:
        cleaned_news = clean_text(news)
        vectorized_input = vectorizer.transform([cleaned_news])
        prediction = model.predict(vectorized_input)[0]
        if prediction == 0:
            st.error("This news is likely **Fake**.")
        else:
            st.success("This news appears to be **Real**.")

accuracy = accuracy_score(y_test, model.predict(X_test))
st.write(f"Model Accuracy: {accuracy:.2f}")
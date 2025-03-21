import streamlit as st
import subprocess
import pickle
import nltk
import pandas as pd
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import  PorterStemmer 
import re

# LOAD PICKLE FILES
model = pickle.load(open('best_model (1).pkl','rb')) 
vectorizer = pickle.load(open('count_vectorizer.pkl','rb')) 

nltk.download('stopwords')
sw = set(stopwords.words('english'))

# TEXT PREPROCESSING
def text_preprocessing(review):
    txt = TextBlob(review)
    result = txt.correct()
    removed_special_characters = re.sub("[^a-zA-Z]", " ", str(result))
    tokens = removed_special_characters.lower().split()
    stemmer = PorterStemmer()

    cleaned = []
    stemmed = []

    for token in tokens:
        if token not in sw:
            cleaned.append(token)
            
    for token in cleaned:
        token = stemmer.stem(token)
        stemmed.append(token)

    return " ".join(stemmed)

# TEXT CLASSIFICATION
def text_classification(review):
    if len(review) < 1:
        st.write("  ")    
    else:
        with st.spinner("Classification in progress..."):
            cleaned_review = text_preprocessing(review)
            process = vectorizer.transform([cleaned_review]).toarray()
            prediction = model.predict(process)
            p = ''.join(str(i) for i in prediction)
            st.write(review)
        
            if p == 'True':
                st.success("The review entered is Legitimate.")
            if p == 'False':
                st.error("The review entered is Fake.")

# MAIN APP
def main():
    st.title("Fake Review Detection Of E-Commerce Electronic Products Using Machine Learning Techniques")

    st.subheader("Fake Review Classifier")
    url = st.text_input("Enter Review Page URL: ")

    if st.button("Check"):
        with st.spinner("Crawling reviews..."):
            # CALL crawler script as subprocess
            subprocess.call(f"python review_crawler.py {url}", shell=True)

        df = pd.read_csv('reviews1.csv')
        reviews = df['body'].to_list()

        for review in reviews:
            text_classification(review)

if __name__ == '__main__':
    main()



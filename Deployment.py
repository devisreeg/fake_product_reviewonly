# LIBRARIES
import streamlit as st
import pickle
import nltk
import pandas as pd
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
import re
import review_crawler

# LOAD PICKLE FILES
model = pickle.load(open('best_model.pkl', 'rb'))  # make sure to rename the file without space
vectorizer = pickle.load(open('count_vectorizer.pkl', 'rb')) 

# Download stopwords once
nltk.download('stopwords')
sw = set(stopwords.words('english'))

# TEXT PREPROCESSING FUNCTION
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

# TEXT CLASSIFICATION FUNCTION
def text_classification(review):
    if len(review) < 1:
        st.write("Empty review.")    
    else:
        cleaned_review = text_preprocessing(review)
        process = vectorizer.transform([cleaned_review]).toarray()
        prediction = model.predict(process)
        p = ''.join(str(i) for i in prediction)
        st.write(f"**Review:** {review}")
        
        if p == 'True':
            st.success("The review is Legitimate.")
        elif p == 'False':
            st.error("The review is Fake.")
        else:
            st.warning("Unknown classification result.")

# STREAMLIT APP
def main():
    st.title("🛒 Fake Review Detection For E-Commerce Electronics 🧠")
    st.markdown("Using Machine Learning techniques to detect fake product reviews.")

    st.subheader("Enter the Product URL:")
    url = st.text_input("Product URL")

    if st.button("Fetch & Check Reviews"):
        if url:
            with st.spinner("Fetching reviews..."):
                # Assuming review_crawler has a function get_reviews(url) that returns a list of reviews
                reviews = review_crawler.get_reviews(url)
                if not reviews:
                    st.warning("No reviews found or URL might be invalid.")
                else:
                    st.success(f"Fetched {len(reviews)} reviews!")
                    for review in reviews:
                        text_classification(review)
        else:
            st.error("Please enter a valid URL.")

# RUN MAIN
if __name__ == '__main__':
    main()


import streamlit as st
import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Function to remove punctuation
def remove_punc(text):
    pattern = r'[' + string.punctuation + ']'
    return re.sub(pattern, " ", text)

# Function to lowercase text
def lower(text):
    return text.lower()

# Function for tokenization
def tokenization(text):
    return re.split(' ', text)

# Function to remove stopwords
def remove_SW(text):
    sw = nltk.corpus.stopwords.words('english')
    return [item for item in text if item not in sw]

# Function to remove digits
def remove_digits(text):
    return [item for item in text if not item.isdigit()]

# Function to lemmatize text
def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(item) for item in text]

# Function to remove empty tokens
def remove_empty_tokens(text):
    return [item for item in text if item != '']

# Function to remove single letters
def remove_single_letters(text):
    return [item for item in text if len(item) > 1]

# Function to calculate TF-IDF similarity
def tfidf_similarity(text1, text2):
    tfidf_vectorizer = TfidfVectorizer()
    docs = [text1, text2]  # Convert to list
    # Check if both documents contain enough content
    if len(text1.split()) == 0 or len(text2.split()) == 0:
        return 0  # Return 0 similarity if either document is empty
    else:
        tfidf_matrix = tfidf_vectorizer.fit_transform(docs)
        cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
        return cosine_sim[0][0]

# Main function
def main():
    st.title("Similarity Analysis")

    # Input text paragraphs
    text1 = st.text_area("Enter the first text paragraph:", height=150)
    text2 = st.text_area("Enter the second text paragraph:", height=150)

    # Button to calculate similarity
    if st.button("Calculate Similarity"):
        # Perform data preprocessing
        text1_processed = remove_punc(text1)
        text1_processed = lower(text1_processed)
        text1_processed = tokenization(text1_processed)
        text1_processed = remove_SW(text1_processed)
        text1_processed = remove_digits(text1_processed)
        text1_processed = lemmatize(text1_processed)
        text1_processed = remove_empty_tokens(text1_processed)
        text1_processed = remove_single_letters(text1_processed)
        text1_processed = ' '.join(text1_processed)

        text2_processed = remove_punc(text2)
        text2_processed = lower(text2_processed)
        text2_processed = tokenization(text2_processed)
        text2_processed = remove_SW(text2_processed)
        text2_processed = remove_digits(text2_processed)
        text2_processed = lemmatize(text2_processed)
        text2_processed = remove_empty_tokens(text2_processed)
        text2_processed = remove_single_letters(text2_processed)
        text2_processed = ' '.join(text2_processed)

        # Calculate TF-IDF similarity
        similarity_score = tfidf_similarity(text1_processed, text2_processed)

        # Display the similarity score
        st.subheader("Similarity Score")
        st.write(similarity_score)

# Execute the main function
if __name__ == "__main__":
    main()

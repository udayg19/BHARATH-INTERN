import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# TRANSFORM FUNCTION FOR PREPROCESSING
ps = PorterStemmer()


def transform_mails(mails):
    mails = mails.lower()
    mails = nltk.word_tokenize(mails)

    y = []
    for i in mails:
        if i.isalnum():
            y.append(i)

    mails = y[:]
    y.clear()

    for i in mails:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    mails = y[:]
    y.clear()

    for i in mails:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open(r"C:\Users\udayg\Downloads\Uday Intern\vectorizer.pkl", 'rb'))
model = pickle.load(open(r"C:\Users\udayg\Downloads\Uday Intern\model.pkl", 'rb'))

st.title("EMAIL/SMS Spam Classifier")

input_msg = st.text_area("Enter TEXT here")

if st.button('Predict'):
    # 1. PREPROCESS TEXT
    transformed_msg = transform_mails(input_msg)

    # 2. VECTORIZE USING TFIDF
    vector_msg = tfidf.transform([transformed_msg])

    # 3. PREDICT USING MULTINOMIAL NAIVE BAYES MODEL
    result = model.predict(vector_msg)[0]

    # 4. DISPLAY THE OUTPUT AS HAM-SPAM TEXT
    if result == 1:
        st.header("SPAM TEXT")
    else:
        st.header("NOT A SPAM TEXT - HAM TEXT")

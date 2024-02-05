import streamlit as st
import pickle 
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


tfid = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

def transform_txt(text):
    
    #lowering the text
    text= text.lower()
    
    #tokenizing
    text = nltk.word_tokenize(text)
    
    # removing special chars
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text = y[:]
    y.clear()
    
    # removing stopwords and punctions
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    # stemmimg
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)
    

st.title("Email/SMS Spam Classifier")

input_sms = st.text_input("Enter the Message")

if st.button('Predict'):
    transformed_sms = transform_txt(input_sms)

    vector_input = tfid.transform([transformed_sms])

    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
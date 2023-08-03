import streamlit as st
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


@st.cache_data
def main():
    train = pd.read_csv(r"streamlit_for_NLP_project\final_train.csv")
    train = train.drop_duplicates()
    train = train.dropna()
    X = train['message']
    Y = train['Language']
    keyword = []
    for text in X:
        text = re.sub(r'[!@#$(),%^*?:;~`0-9]', ' ', text)
        text = text.lower()
        keyword.append(text)
    cv = CountVectorizer()
    X = cv.fit_transform(keyword)
    le = LabelEncoder()
    Y = le.fit_transform(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    model = LinearSVC().fit(X_train, Y_train)

    return model, cv, le

# stremlit code
st.set_page_config(page_title='Multilingual Language Translation System', page_icon='📚', layout='centered', initial_sidebar_state='auto')
st.title('Multilingual Language System')
st.spinner('Loading...')


model,cv ,le = main()

#Language Sentence input
text = st.text_input('Enter The Text -->',label_visibility="visible", disabled=False, max_chars=None, key=None, type='default', value='hi')
button = st.button('Predict')
if button:
    if text == '':
        st.write("Please enter the text")
    else:
        qt = []
        qt.append(text)
        with st.spinner('Predicting...'):
            qt = cv.transform(qt)
            res = le.inverse_transform(model.predict(qt))
            st.info(res[0])

from turtle import title
import pandas as pd
import streamlit as st 
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
raw_data = pd.read_csv('train.csv')

raw_data = raw_data.fillna('')
port = PorterStemmer()  
vectorizer = TfidfVectorizer()

X = raw_data['title'].values
y = raw_data['label'].values

vectorizer.fit(X)
X = vectorizer.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

lm = LogisticRegression()
lm.fit(X_train,y_train)
predictions = lm.predict(X_test)

def welcome():
    return "Welcome All"

def fake_news_det(news):
    input_data = [news]
    vectorized_input_data = vectorizer.transform(input_data)
    prediction = lm.predict(vectorized_input_data)
    return prediction


def main():
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h1 style="color:white;text-align:center;">Fake News Prediction ML App </h1>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.markdown("""---""")
    user_data = st.text_input("Type Here","Enter News")
    result=""
    op=""
    st.markdown("""---""")
    if st.button("Predict"):
        result=fake_news_det(user_data)
        if result == [1]:
            op = "Fake"
        else:
            op = "Real"
    st.markdown("""---""")
    st.success('The News is {}'.format(op))

if __name__=='__main__':
    main()



# Fake : Anonymous Donor Pays $2.5 Million To Release Everyone Arrested At The Dakota Access Pipeline
# Real : A Back-Channel Plan for Ukraine and Russia, Courtesy of Trump Associates - The New York Times
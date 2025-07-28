# train_model.py
import pandas as pd

# Load dataset
df = pd.read_csv('career_guidance_dataset.csv')

# Rename columns if necessary
df.columns = ['Role', 'Question', 'Answer']
print(df.head())


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import string

#clean text
def preprocess(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text

df['cleaned_question'] = df['Question'].apply(preprocess)

#Tf-idf vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_question'])
y = df['Role']


# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#Evaluete 
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the model and vectorizer
joblib.dump(model, 'career_guidance_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')


#app.py
import streamlit as st
import joblib

#Load the model and vectorizer
model = joblib.load('career_guidance_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

st.title("Career Guidance Chatbot")
st.markdown("Ask me about different career roles, and I'll suggest the best fit for you!")

user_input = st.text_input("Enter your question about career roles:")

if user_input:
    cleaned = user_input.lower()
    vectorized_input = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized_input)
    st.success(f"Suggested Career Role: **{prediction[0]}**")
import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load the pre-trained model and the saved TF-IDF vectorizer
model_path = 'C:/Users/HP/Downloads/Online Retail/notebooks/logistic_regression_model.joblib'
tfidf_path = 'C:/Users/HP/Downloads/Online Retail/notebooks/tfidf_vectorizer.joblib'

model = joblib.load(model_path)
tfidf = joblib.load(tfidf_path)

# Preprocess function
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Join tokens back to a single string
    return ' '.join(tokens)

# Streamlit app interface
st.title("Online Retail Review Sentiment Predictor")
st.write("Enter a review to predict if it is positive or negative.")

# Text input from the user
user_input = st.text_area("Enter your review here:")

# Process and predict
if st.button("Predict"):
    if user_input:
        # Preprocess the input
        cleaned_input = preprocess_text(user_input)
        # Transform the input using the loaded TF-IDF vectorizer
        input_tfidf = tfidf.transform([cleaned_input])
        # Predict the sentiment
        prediction = model.predict(input_tfidf)[0]
        # Display the prediction
        if prediction == 1:
            st.write("The review sentiment is **positive**.")
        else:
            st.write("The review sentiment is **negative**.")
    else:
        st.write("Please enter a review to analyze.")

import streamlit as st
from predict import predict_sentiment  # Place your predict_sentiment function in predict.py

st.title("Movie Review Sentiment Analysis")

user_input = st.text_area("Enter a movie review:", "")
if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a valid review.")
    else:
        sentiment, confidence = predict_sentiment(user_input)
        color = "🟢" if sentiment == "Positive" else "🔴"
        st.markdown(f"{color} **{sentiment} Sentiment** (Confidence: {confidence})")

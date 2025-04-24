import streamlit as st
from transformers import pipeline
from transformers import pipeline

sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", framework="pt")


# Load Sentiment Analysis Model
MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
sentiment_analyzer = pipeline("sentiment-analysis", model=MODEL_NAME)

# Streamlit App UI
st.title("🔍 Sentiment Analysis App")
st.write("Analyze the sentiment of text using DistilBERT.")

# User Input
user_input = st.text_area("Enter text for sentiment analysis:", "")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        result = sentiment_analyzer(user_input)
        sentiment = result[0]['label']
        confidence = result[0]['score']

        # Display Results
        st.subheader("📌 Sentiment Analysis Result:")
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Confidence Score:** {confidence:.2f}")

        # Add Emoji Feedback
        if sentiment == "POSITIVE":
            st.success("✅ Positive Sentiment! 😀")
        elif sentiment == "NEGATIVE":
            st.error("❌ Negative Sentiment! 😡")
        else:
            st.warning("⚠️ Neutral Sentiment! 😐")
    else:
        st.warning("Please enter some text for analysis.")

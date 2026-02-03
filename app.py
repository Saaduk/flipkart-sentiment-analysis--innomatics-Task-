import streamlit as st
import pickle

# Load trained model
model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

st.set_page_config(page_title="Flipkart Sentiment Analysis")

st.title("🛒 Flipkart Review Sentiment Analyzer")
st.write("Enter a product review to analyze its sentiment.")

review = st.text_area("✍️ Write your review here")

if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review")
    else:
        review_vec = vectorizer.transform([review])
        prediction = model.predict(review_vec)[0]

        if prediction == 1:
            st.success("Positive Review 😊")
        elif prediction == 0:
            st.info("Neutral Review 😐")
        else:
            st.error("Negative Review 😠")

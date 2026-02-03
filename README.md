# Flipkart Product Review Sentiment Analysis

This project analyzes customer reviews from Flipkart and predicts whether a review is Positive, Neutral, or Negative using Machine Learning.

## Tech Stack
- Python
- Pandas
- Scikit-learn
- Streamlit

## Dataset
Flipkart Reviews Dataset from Kaggle

## Sentiment Logic
- Rating >= 4 → Positive
- Rating = 3 → Neutral
- Rating <= 2 → Negative

## How to Run Locally
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py

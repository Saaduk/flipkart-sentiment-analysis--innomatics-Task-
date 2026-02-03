import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Paths
DATA_PATH = "data/flipkart_reviews.csv"
MODEL_DIR = "model"

# Create model directory if not exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Check column names (important)
print(df.columns)

# Rename columns if needed (adjust if your column names differ)
# Common names: review, review_text, reviews
if 'review' in df.columns:
    text_col = 'review'
elif 'review_text' in df.columns:
    text_col = 'review_text'
else:
    text_col = df.columns[0]  # fallback

rating_col = 'rating'

df = df[[text_col, rating_col]]
df.dropna(inplace=True)

# Convert rating to sentiment
def get_sentiment(rating):
    if rating >= 4:
        return 1   # Positive
    elif rating == 3:
        return 0   # Neutral
    else:
        return -1  # Negative

df['sentiment'] = df[rating_col].apply(get_sentiment)

X = df[text_col]
y = df['sentiment']

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_vec = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_vec, y)

# Save model and vectorizer
pickle.dump(model, open(f"{MODEL_DIR}/model.pkl", "wb"))
pickle.dump(vectorizer, open(f"{MODEL_DIR}/vectorizer.pkl", "wb"))

print("✅ Model training completed and files saved.")

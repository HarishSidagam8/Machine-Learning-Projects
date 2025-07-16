import pandas as pd
import re
import nltk
import joblib
import logging
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocess.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logging.info("🚀 Starting preprocessing...")

# Download only stopwords
nltk.download('stopwords')

# Load dataset
try:
    df = pd.read_csv("spotify_millsongdata.csv").sample(10000)
    logging.info("✅ Dataset loaded and sampled: %d rows", len(df))
except Exception as e:
    logging.error("❌ Failed to load dataset: %s", str(e))
    raise e

df = df.drop(columns=['link'], errors='ignore').reset_index(drop=True)

# Clean text
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))  # Remove non-alphabetic
    text = text.lower()
    tokens = text.split()  # Simpler tokenizer, avoids NLTK dependency
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

logging.info("🧹 Cleaning text...")
df['cleaned_text'] = df['text'].apply(preprocess_text)
logging.info("✅ Text cleaned.")

# TF-IDF
logging.info("🔠 Vectorizing using TF-IDF...")
tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])
logging.info("✅ TF-IDF matrix shape: %s", tfidf_matrix.shape)

# Cosine similarity
logging.info("📐 Calculating cosine similarity...")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
logging.info("✅ Cosine similarity matrix generated.")

# Save outputs
joblib.dump(df, 'df_cleaned.pkl')
joblib.dump(tfidf_matrix, 'tfidf_matrix.pkl')
joblib.dump(cosine_sim, 'cosine_sim.pkl')
logging.info("💾 Data saved to disk.")
logging.info("✅ Preprocessing complete.")


# %%
import pickle
import numpy as np
import tensorflow as tf

# %%
import os
BASE_DIR = r"D:\AI_PROJECTS\Sentiment Analysis"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# %%
# Load TF-IDF
with open(f"{MODEL_DIR}/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

# Load Label Encoder
with open(f"{MODEL_DIR}/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Load ML models
with open(f"{MODEL_DIR}/decision_tree.pkl", "rb") as f:
    dt = pickle.load(f)

with open(f"{MODEL_DIR}/naive_bayes.pkl", "rb") as f:
    nb = pickle.load(f)

with open(f"{MODEL_DIR}/xgboost.pkl", "rb") as f:
    xgb = pickle.load(f)

# Load LSTM
with open(f"{MODEL_DIR}/lstm/tokenizer.pkl", 'rb') as f:
    tokenizer = pickle.load(f)

lstm_model = tf.keras.models.load_model(
    f"{MODEL_DIR}/lstm/lstm_model.h5"
)


# %%
# Text cleaning (same as training)
# ------------------------------
import re
import string

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text



# %%
# Majority Vote Function
from collections import Counter

def majority_vote(preds):
    c = Counter(preds)
    return c.most_common(1)[0][0]


# %%
def predict_sentiment(text):

    cleaned = clean_text(text)

    # 1) ML pipelines
    tfidf_vec = tfidf.transform([cleaned])

    p_dt  = dt.predict(tfidf_vec)[0]
    p_nb  = nb.predict(tfidf_vec)[0]
    p_xgb = xgb.predict(tfidf_vec)[0]

    # 2) LSTM pipeline
    seq = tokenizer.texts_to_sequences([cleaned])
    seq = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=120)
    p_lstm = np.argmax(lstm_model.predict(seq)[0])

    # Majority vote
    final = majority_vote([p_dt, p_nb, p_xgb, p_lstm])

    # Convert to label
    return le.inverse_transform([final])[0]

# %%




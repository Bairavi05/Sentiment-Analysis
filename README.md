# 🎯 Sentiment Analysis (ML + LSTM + Ensemble)  

A complete sentiment analysis system that predicts **six emotions** — sad, joy, love, anger, fear, surprise — using a hybrid model combining **Machine Learning**, **LSTM**, and **Ensemble Voting**. The project includes a ready-to-use **Streamlit web app**.

---

## 🚀 Features  
- Text preprocessing + cleaning  
- ML models (TF-IDF based): Decision Tree, Naive Bayes, XGBoost  
- Deep Learning model: LSTM (Keras/TensorFlow)  
- Ensemble voting for final prediction  
- Streamlit UI for easy interaction  

---

## 📁 Project Structure  

Sentiment Analysis/
│
├── app/
│ └── app.py # Streamlit UI
│
├── engine/
│ └── engine.py # Prediction engine (ML + LSTM + Ensemble)
│
├── models/
│ ├── tfidf_vectorizer.pkl
│ ├── label_encoder.pkl
│ ├── decision_tree.pkl
│ ├── naive_bayes.pkl
│ ├── xgboost.pkl
│ └── lstm/
│ ├── tokenizer.pkl
│ └── lstm_model.h5
│
├── requirements.txt
└── README.md


---

## 🧠 How Prediction Works  
For every input text:

1. Text is cleaned  
2. TF-IDF → ML models (DT, NB, XGB)  
3. Tokenizer → LSTM model  
4. All 4 predictions are combined  
5. Majority vote decides the final emotion  

---

## ▶️ Run Locally  

### 1️⃣ Install dependencies  
pip install -r requirements.txt


### 2️⃣ Run Streamlit app  
streamlit run app/app.py


---

## 🌐 Deployment  
This project is fully compatible with **Streamlit Cloud**:

- Upload the repository to GitHub  
- Connect the repo to Streamlit Cloud  
- Deploy instantly  

---

## 🏗 Technologies Used  
- Python  
- TensorFlow / Keras  
- Scikit-Learn  
- XGBoost  
- NumPy, Pandas  
- Streamlit  

---

## ✨ Output Labels with Emojis  
| Label     | Emoji |
|-----------|--------|
| sad       | 😢 |
| joy       | 😊 |
| love      | ❤️ |
| anger     | 😡 |
| fear      | 😨 |
| surprise  | 😲 |

---

## 📌 Future Improvements  
- Add BERT / DistilBERT for better accuracy  
- Add real-time Twitter/YouTube comment scrapers  
- Add database storage for predictions  

---

## 📬 Author  
**Bairavi (AI & Data Science)**  
Sentiment Analysis Project — 2025  


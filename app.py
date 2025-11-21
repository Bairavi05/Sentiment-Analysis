import streamlit as st
import sys
import os


from engine import predict_sentiment   # <-- your function

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Sentiment Analysis", layout="centered")

st.title("📊 Sentiment Analysis — ML + LSTM Ensemble")
st.write("This app predicts sentiment using **Decision Tree**, **Naive Bayes**, "
         "**XGBoost**, and **LSTM**, combined using **majority voting**.")

# Input box
user_text = st.text_area("Enter your text here:", height=200)

# Predict button
if st.button("Analyze"):
    if user_text.strip() == "":
        st.warning("Please enter some text!")
    else:
        with st.spinner("Analyzing..."):
            result = predict_sentiment(user_text)
        
        str_res=''
        if result==0:
            str_res='Sadness 😢'
        elif result==1:
            str_res='Joy 😄'
        elif result==2:
            str_res='Love ❤️'
        elif result==3:
            str_res='Anger 😡'
        elif result==4:
            str_res='Fear 😨'
        else:
            str_res='Surprise 😮'
        st.success(f"Predicted Sentiment: **{str_res}**")

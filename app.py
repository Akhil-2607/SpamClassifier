import streamlit as st
import joblib

# Load model + vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Title
st.title("📬 Spam vs Ham Classifier")
st.write("This Model predicts whether the message is Spam or Ham.")

# Input
user_input = st.text_area("Type your message here:")

# Predict button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        input_vec = vectorizer.transform([user_input])
        pred = model.predict(input_vec)[0]
        prob = model.predict_proba(input_vec)[0][pred]
        Category = " Spam" if pred == 1 else " Not Spam"
        
        st.subheader(f"Prediction: {Category}")
        st.write(f"Confidence: {prob:.2f}")
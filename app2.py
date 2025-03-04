import streamlit as st
from groq import Groq
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

api_key = st.secrets["API_KEY"]

client = Groq(api_key=api_key)

data = pd.read_csv("/workspaces/GROQ-llama3.1/nlp_intent_dataset.csv")
X= data["User Query"]
Y = data["Intent"]

pipeline = Pipeline(
    [("tfidf",TfidfVectorizer()),
    ("clf",MultinomialNB())]
)
pipeline.fit(X,Y)

# Intent Response Mapping
responses = {
    "Password_Reset": "To reset your password, go to settings and click 'Forgot Password'.",
    "Check_Balance": "Your current account balance is $5000.",
    "Order_Cancellation": "You can cancel your order from 'My Orders' section.",
    "Order_Status": "Your order is being processed. Check your email for updates."
}

def predict_intent(userip):
    return pipeline.predict([userip])[0]

def chatbot_response(text):
    intent = predict_intent(text)

    if intent in responses:
        return responses[intent]
    else:
        return llama_response(text)

# Write a function to get response from the Groq client
def llama_response(text: str):
    stream = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assitant"
            },
            {
                "role": "user",
                "content": text
            }
        ],
        model = "llama-3.3-70b-versatile",
        stream= True
    )

    for chunk in stream:
        reponse = chunk.choices[0].delta.content
        if reponse is not None:
            yield reponse

# Start building streamlit app
st.title("Llama 3.3 Model")
st.subheader("by Sindhura Nadendla")

text = st.text_area("Please ask any question : ")

if text:
    st.subheader("Model Response : ")
    #st.write_stream(llama_response(text))

# Get response from the chatbot (either predefined or AI-generated)
    response = chatbot_response(text)
    
    # Handle both static and streamed responses
    if isinstance(response, str):
        st.write(response)  # Predefined response
    else:
        st.write_stream(response)  # LLaMA-generated response
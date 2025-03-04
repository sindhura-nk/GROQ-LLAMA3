# import necessary libraries
import streamlit as st
from groq import Groq

# retrieve api key details from secret file
api_key = st.secret["API_KEY"]

# create a groq instance
client = Groq(api_key=api_key)

# Load the data for NLP Intent Recognition
df = 
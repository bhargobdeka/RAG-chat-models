import requests
import streamlit as st

import os
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
## Langsmith tracking
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY") # where monitoring results needs to be stored

def get_openai_response(input_text):
    response = requests.post("http://localhost:8000/stock-trader/invoke",
    json={'input':{'input':input_text}})
    return response.json()['output']['output']


## streamlit framework
st.title('Search for Stocks')
input_text = st.text_input('Enter your stock-related question')

if input_text:
    response = get_openai_response(input_text)
    st.write(response)
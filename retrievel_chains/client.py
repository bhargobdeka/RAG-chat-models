import requests
import streamlit as st
from langchain.schema.runnable import RunnableMap
from langserve import RemoteRunnable


def get_openai_response(input_text):
    response=requests.post("http://localhost:8000/budget_chain/invoke",
    json={'input':input_text})

    return response.json()

st.title('Canada Budget 2024')
input_text = st.text_input('ask budget related question here')

if input_text:
    st.write(get_openai_response(input_text))


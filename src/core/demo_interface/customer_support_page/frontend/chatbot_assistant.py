"""
 Chatbot assistant main page definition
"""
import json
import pandas as pd
import streamlit as st
from core.demo_interface.customer_support_page.chatbot_assistant_functions import chatbot_response
from core.utils.data_modification import (
    read_data,
)
from datetime import datetime

__all__ = [
    'chatbot_assistant_main'
]

def chatbot_assistant_main(sesh):
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    st.title("Welcome to ADP virtual assistant")

    # Input box for the user to type messages
    user_input = st.text_input("Please write your inquiry:", key="input")

    if st.button("Request"):
        if user_input:
            # Store the user message
            response = chatbot_response(user_input, model_name='model_best')
            st.write(response)
            st.session_state['messages'].append(user_input)


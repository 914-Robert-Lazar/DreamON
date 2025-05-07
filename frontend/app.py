import streamlit as st
import random
import time
import requests

# To be changed for backend call
def get_response(prompt):
    url = "http://localhost:8000/process_query"
    payload = {"query": prompt}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json().get("content", "Sorry, I couldn't process your request.")
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}"

st.title("DreamON: a chatbot to explain your most precious dreams")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What did you dream last night?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write(get_response(prompt=prompt))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
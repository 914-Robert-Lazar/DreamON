import streamlit as st
import random
import time
import requests
import json

# Regular non-streaming response
def get_response(prompt):
    url = "http://localhost:8000/process_query"
    payload = {"query": prompt}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json().get("content", "Sorry, I couldn't process your request.")
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}"

# Streaming response function
def get_streaming_response(prompt, placeholder):
    url = "http://localhost:8000/process_query"
    payload = {"query": prompt}
    full_response = ""
    
    try:
        with requests.post(url, json=payload, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data:'):
                        if line.strip() == 'data: [DONE]':
                            break
                        data = line[5:].strip()  # Remove 'data: ' prefix
                        try:
                            chunk = json.loads(data)
                            if 'content' in chunk:
                                full_response += chunk['content']
                                placeholder.markdown(full_response)
                        except json.JSONDecodeError:
                            pass
        
        return full_response
    except requests.exceptions.RequestException as e:
        placeholder.markdown(f"An error occurred: {e}")
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
        st.markdown(prompt)    # Display assistant response in chat message container with streaming
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        # Add a loading indicator while waiting for response
        with st.spinner("Thinking..."):
            # Use get_response without the placeholder parameter
            response = get_response(prompt=prompt)
            # Display the response
            response_placeholder.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
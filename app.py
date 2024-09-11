from chatbot import RAGChatbot
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, timezone


@st.cache_data
def load_data():
    df = pd.read_csv('data/reddit_data.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])  
    return df


@st.cache_data
def load_embeddings(filtered_df):
    chatbot = RAGChatbot(filtered_df, api_key = st.secrets["HUGGINGFACE_API_KEY"])
    return chatbot
    

df = load_data()

months =  st.select_slider(
    "Select how many months of Reddit posts to scrape",
    options = [3, 6, 9, 12] 
    )

end_date = datetime.now(timezone.utc)
start_date = end_date-timedelta(days=30 * months)
filtered_df = df[df['Timestamp'] > start_date]

chatbot = load_embeddings(filtered_df)

st.title("NTU Subreddit Chatbot")

# Store LLM generated responses
if "messages" not in st.session_state():
    st.session_state.messages = [{"role": "assistant", "content": "Welcome! Please ask any questions regarding student life at NTU :)"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if input := st.chat_input():
    if input.strip():
        st.session_state.messages.append({"role": "user", "content": input})
        with st.chat_message("user"):
            st.write(input)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner("Working on your answer..."):
            try:
                response = chatbot.ask_question(input)
            except Exception as e:
                response = "Sorry, an error occurred while processing your request. Please try again."
                st.error(f"Error: {str(e)}")
            
            st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
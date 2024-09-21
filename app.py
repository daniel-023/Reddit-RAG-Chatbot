from chatbot import RAGChatbot
import streamlit as st
import pandas as pd
import faiss
from datetime import datetime, timedelta, timezone


@st.cache_data
def load_data():
    df = pd.read_csv('data/reddit_data.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])  
    return df


@st.cache_resource
def load_index(index_path:str):
    index = faiss.read_index(index_path)
    return index

@st.cache_resource
def load_chatbot(df, _index):
    chatbot = RAGChatbot(api_key=st.secrets["HUGGINGFACE_API_KEY"], df=df, index=index)
    return chatbot


# @st.cache_resource
# def load_embeddings(df):
#     
#     chatbot.load_df(filtered_df)
#     chatbot.build_vector_db()
#     return chatbot


df = load_data()
index = load_index('data/faiss_index.index')
chatbot = load_chatbot(df=df, _index=index)


months =  st.select_slider(
    "Choose the time range for Reddit posts (in months):",
    options = [3, 6, 9, 12] 
    )

end_date = datetime.now(timezone.utc)
start_date = end_date-timedelta(days=30 * months)
filtered_df = df[df['Timestamp'] > start_date]


st.title("NTU Subreddit Chatbot")

# Store LLM generated responses
if "messages" not in st.session_state:
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

        with st.chat_message("assistant"):
            with st.spinner("Working on your answer..."):
                response = chatbot.generate_answer(input)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.write(response.replace('\n', ' '))
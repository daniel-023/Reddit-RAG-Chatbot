from chatbot import RAGChatbot
import streamlit as st
import pandas as pd
import faiss


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
    chatbot = RAGChatbot(api_key=st.secrets["HUGGINGFACE_API_KEY"], df=df, index=_index)
    return chatbot



df = load_data()
index = load_index('data/faiss_index.index')
chatbot = load_chatbot(df=df, _index=index)


st.title("NTU Subreddit Chatbot")


sample_qns = [
    "How is life at NTU for undergraduates?",
    "What do students say about NTU's housing facilities?",
    "What are common academic concerns for NTU students?",
    "What do students feel about their food options in NTU?"
]


with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This chatbot uses Retrieval-Augmented Generation (RAG) to analyse discussions from the Nanyang Technological University (NTU) subreddit. 
        Ask questions about student life, academic concerns, housing, social activities, and more. 
        The chatbot provides concise, specific responses by quoting posts and comments from the subreddit over the past year. 
        """
    )

st.subheader("Sample Questions")
for question in sample_qns:
    if st.button(question):
        st.session_state.clicked_question = question


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome! Please ask any questions regarding student life at NTU :)"}]


# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# Process clicked sample question
if "clicked_question" in st.session_state:
    input = st.session_state.clicked_question
    del st.session_state.clicked_question  # Clear the clicked question after processing
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

    with st.chat_message("assistant"):
        with st.spinner("Working on your answer..."):
            response = chatbot.generate_answer(input)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response.replace('\n', ' '))


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


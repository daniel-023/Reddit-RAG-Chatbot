from pathlib import Path
import os
import platform

# Runtime stability defaults for macOS + PyTorch/SentenceTransformers.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import streamlit as st

from chatbot import RAGChatbot


DATA_PATH = Path("data/reddit_data.csv")
INDEX_PATH = Path("data/faiss_index.index")
METADATA_PATH = Path("data/faiss_metadata.csv")
EMBEDDINGS_PATH = Path("data/embeddings.npy")

DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

SAMPLE_QUESTIONS = [
    "What do students say about the social life at NTU?",
    "What do students feel about their mental health?",
    "What do NTU students feel about the new Flexible Grading Option (FGO)?",
    "What do students say about NTU's housing facilities?",
    "What do university students feel about campus transport?",
    "How are NTU's food options?",
    "What are some popular study spots at NTU?",
]


@st.cache_data
def load_data(data_path: str):
    df = pd.read_csv(data_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    return df


@st.cache_data
def load_chunk_metadata(path: str):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None


@st.cache_resource
def load_index(index_path: str):
    import faiss
    return faiss.read_index(index_path)


@st.cache_resource
def load_embeddings_matrix(path: str):
    return np.load(path)


@st.cache_resource
def load_chatbot(_df, _index_obj, llm_repo_id, _chunk_metadata_df, api_key):
    return RAGChatbot(
        api_key=api_key,
        df=_df,
        index=_index_obj,
        llm_repo_id=llm_repo_id,
        chunk_metadata_df=_chunk_metadata_df,
    )


def render_sources(sources):
    if not sources:
        return
    with st.expander("Sources", expanded=False):
        for i, source in enumerate(sources, start=1):
            title = source.get("title", "Untitled")
            timestamp = source.get("timestamp", "Unknown time")
            url = source.get("url")
            if url:
                st.markdown(f"{i}. [{title}]({url}) ({timestamp})")
            else:
                st.markdown(f"{i}. {title} ({timestamp})")
            st.caption(source.get("snippet", ""))


def render_chat_message(message):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))


def append_and_render_assistant_reply(chatbot, user_input):
    with st.chat_message("assistant"):
        with st.spinner("Working on your answer..."):
            try:
                result = chatbot.generate_answer_with_sources(user_input)
                response = result.get("answer", "I don't know based on the retrieved Reddit comments.")
                sources = result.get("sources", [])
            except Exception as exc:
                response = f"I hit an error while generating a response: {exc}"
                sources = []

            message = {"role": "assistant", "content": response, "sources": sources}
            st.session_state.messages.append(message)
            st.write(response.replace("\n", " "))
            render_sources(sources)


is_macos = platform.system() == "Darwin"
required_index_file = EMBEDDINGS_PATH if is_macos else INDEX_PATH
missing_files = [str(path) for path in [DATA_PATH, required_index_file] if not path.exists()]
if missing_files:
    st.error(f"Missing required data files: {', '.join(missing_files)}")
    st.info("Generate artifacts first: `python build_df.py` then `python generate_index.py`.")
    st.stop()

api_key = st.secrets.get("HUGGINGFACE_API_KEY") or os.getenv("HUGGINGFACE_API_KEY")
if not api_key:
    st.error("Missing Hugging Face API key.")
    st.info("Set `HUGGINGFACE_API_KEY` in `.streamlit/secrets.toml` or environment variables.")
    st.stop()

try:
    df = load_data(str(DATA_PATH))
    chunk_metadata_df = load_chunk_metadata(str(METADATA_PATH))
    if is_macos:
        index = load_embeddings_matrix(str(EMBEDDINGS_PATH))
    else:
        index = load_index(str(INDEX_PATH))
except Exception as exc:
    st.error(f"Failed to load app data/index: {exc}")
    st.stop()

chatbot = load_chatbot(
    _df=df,
    _index_obj=index,
    llm_repo_id=DEFAULT_MODEL,
    _chunk_metadata_df=chunk_metadata_df,
    api_key=api_key,
)

latest_timestamp = df["Timestamp"].max()
if pd.notna(latest_timestamp):
    freshness_sgt = latest_timestamp.tz_convert("Asia/Singapore").strftime("%Y-%m-%d %H:%M SGT")
else:
    freshness_sgt = "Unknown"

st.title("NTU Subreddit Chatbot")
st.caption(f"Data freshness: {freshness_sgt}")

with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This chatbot uses Retrieval-Augmented Generation (RAG) to analyse conversations in the
        Nanyang Technological University (NTU) subreddit.
        """
    )
    if chunk_metadata_df is None:
        st.warning("`data/faiss_metadata.csv` not found. Sources may be less precise.")
    if st.button("Clear chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Welcome! Please ask any questions regarding student life at NTU :)", "sources": []}
        ]
        st.rerun()

st.subheader("Sample Questions")
for i, question in enumerate(SAMPLE_QUESTIONS):
    if st.button(question, key=f"sample_q_{i}"):
        st.session_state.clicked_question = question

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome! Please ask any questions regarding student life at NTU :)", "sources": []}
    ]

for message in st.session_state.messages:
    render_chat_message(message)

if "clicked_question" in st.session_state:
    user_input = st.session_state.clicked_question
    del st.session_state.clicked_question
    user_message = {"role": "user", "content": user_input}
    st.session_state.messages.append(user_message)
    render_chat_message(user_message)
    append_and_render_assistant_reply(chatbot, user_input)

if user_input := st.chat_input("Ask about NTU student life..."):
    if user_input.strip():
        user_message = {"role": "user", "content": user_input}
        st.session_state.messages.append(user_message)
        render_chat_message(user_message)
        append_and_render_assistant_reply(chatbot, user_input)

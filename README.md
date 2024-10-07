# r/NTU Retrieval-Augmented Generation (RAG) Chatbot

> [!WARNING]
> The LLM may generate hallucinations or misinformation.

## Table of Contents
- [Introduction](#introduction)
- [Tech Stack](#tech-stack)
- [References](#references)
## Introduction
This project aims to index conversations in the Nanyang Technological University's (NTU) subreddit, enabling outsiders to gain insight into the daily lives of university students.

By interacting with the chatbot, users can understand how students feel about undergraduate housing, campus transport, academic policies, and more.

## Tech Stack
### Data Collection
[PRAW (Python Reddit API Wrapper)](https://github.com/praw-dev/praw) 
### Vector Database
[FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss)
[Sentence Transformers](https://www.sbert.net/)
### RAG Pipeline
[LangChain](https://github.com/hwchase17/langchain)
[Hugging Face Transformers](https://huggingface.co/transformers/)
### App Framework
[Streamlit](https://streamlit.io/) 
### Data Automation
[GitHub Actions](https://github.com/features/actions)
  
## References
[Scraping Reddit using Python](https://www.geeksforgeeks.org/scraping-reddit-using-python/)

[Build a Simple RAG Chatbot with LangChain](https://medium.com/credera-engineering/build-a-simple-rag-chatbot-with-langchain-b96b233e1b2a)

[How to Build a Simple RAG-based LLM Chatbot](https://medium.com/@turna.fardousi/how-to-build-a-simple-rag-llm-chatbot-47f3fcec8c85)

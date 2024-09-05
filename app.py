import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import gradio as gr

load_dotenv()


class RAGChatbot:
    def __init__(self, csv_file: str, model_name: str = 'all-MiniLM-L6-v2'):
        # Load the CSV file
        self.data = pd.read_csv(csv_file)
        
        # Initialise embedding model
        self.embedding_model = SentenceTransformer(model_name)
        self.embeddings = HuggingFaceEmbeddings(model_name = model_name)

        # Create documents
        self.documents = self._create_documents()

        # Create FAISS vector store
        self.vector_store = FAISS.from_texts(self.documents, self.embeddings)

        # Initialise Langchain RAG pipeline
        self.llm = HuggingFaceEndpoint(
            repo_id= "mistralai/Mistral-7B-Instruct-v0.3", 
            temperature = 0.8, 
            top_k = 50, 
            huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
            )

        self.prompt = PromptTemplate(input_variables=["context", "question"],
            template= """
            Your job is to study the student discussions in the subreddit for Nanyang Technological University (NTU). 
            You wil answer questions about the student's discussions, their concerns, and other questions pertaining to student life at NTU.
            Use the following context to answer the question. 
            If you don't know the answer, just say you don't know. 
            Keep the answer within 3 paragraphs and concise.

            Context: {context}
            Question: {question}
            Answer:
        """
        )

        self.rag_chain = (
        {"context": self.vector_store.as_retriever(), "question": RunnablePassthrough()}
        | self.prompt
        | self.llm
        | StrOutputParser()
    )

    def _create_documents(self):
        return(self.data['Title'] + ". " + self.data['Text']).tolist()
    

chatbot = RAGChatbot('data/subreddit_comments.csv')  
input = "Were there any discussions on the Israel-Hamas conflict?"   
result = chatbot.rag_chain.invoke(input)
print(result)

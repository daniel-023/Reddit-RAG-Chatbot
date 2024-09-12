import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser


class RAGChatbot:
    def __init__(self, df, model_name = 'all-MiniLM-L6-v2', api_key=None):

        # Initialise embedding model
        self.embedding_model = SentenceTransformer(model_name)
        self.index = None
        self.data = df
        self.api_key = api_key
        # Initialise Langchain RAG pipeline
        self.llm = self._initialize_llm()

        self.prompt = PromptTemplate(input_variables=["context", "question"],
            template= """
            Your job is to study the student discussions in the subreddit for Nanyang Technological University (NTU). 
            You wil answer questions about the students' discussions, their concerns, and other questions pertaining to student life at NTU.
            Be as specific as possible, use the following context to answer the question. 
            If you don't know the answer, just say you don't know. 
            Keep the answer within 3 paragraphs and concise.

            Context: {context}
            Question: {question}
            Answer:
            """
        )

    def load_data(self, data):
        self.data = data
        self._generate_embeddings()

    def _generate_embeddings(self):
        if self.data is not None:
            # Ensure there are no null values in 'Title' or 'Text'
            self.data.fillna({'Title': 'No Title'}, inplace=True)
            self.data.fillna({'Text': 'No Text'}, inplace=True)
            texts = self.data['Title'] + ": " + self.data['Text']
            embeddings = self.embedding_model.encode(texts.tolist(), convert_to_numpy=True)
            self.index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance: Euclidean distance
            self.index.add(embeddings)

        else:
            raise ValueError("No data loaded to generate embeddings.")


    def query_faiss(self, query, top_k = 5):
        query_embedding = self.embedding_model.encode([query])[0]
        distances, indices = self.index.search(np.array([query_embedding]), top_k)
        relevant_rows = self.data.iloc[indices[0]]
        return relevant_rows, distances            


    def generate_answer(self, relevant_rows, question):
        relevant_contexts = "\n".join(
            (relevant_rows['Title'] + ": " + relevant_rows['Text']).tolist()
        )

        inputs = {
            "context": relevant_contexts,
            "question": question
        }

        rag_chain = (
            self.prompt 
            | self.llm
            | StrOutputParser()
        )
        return rag_chain.invoke(inputs)
    

    def ask_question(self, question):
        if self.index is None:
            raise ValueError("FAISS index is not initialised. Please load data first")
            
        relevant_rows, distances = self.query_faiss(question, top_k = 5)
        return self.generate_answer(relevant_rows, question)


    def _initialize_llm(self):
        return HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3", 
            temperature=0.4, 
            top_k=35, 
            huggingfacehub_api_token=self.api_key  # Load token from environment variable
        )


# Sample questions:
# What do students feel about the satisfactory/unsatisfactory option for their grades?
# What challenges do NTU university students face?
# What do NTU students discuss in this subreddit?
# What do the university students feel about the campus transport?
# What type of academic questions do the students ask in this subreddit? Be specific about the students' majors and questions asked.
# Tell me more about NTU students' daily lives.
# According to the subreddit, what are the most popular clubs in NTU?
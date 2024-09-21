import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser


class RAGChatbot:
    def __init__(self, api_key, df, index):
        self.documents = []
        self.embeddings = None
        self.index = index
        self.api_key = api_key
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.prompt = PromptTemplate(input_variables=["context", "question"],
            template= """
        Your task is to analyze student discussions in the subreddit for Nanyang Technological University (NTU). 
        You will answer questions about the students' concerns, discussions, and other aspects of student life at NTU.
        When providing your response:
        - Be as specific as possible and **always include direct quotes** from the relevant discussions as evidence for your answers.
        - Use the **exact wording** from the context provided to ensure accuracy.
        - If the context does not provide enough information to answer the question, simply respond with "I don't know."
        - Keep your answers concise and to the point, focusing only on the information directly related to the question.


            Context: {context}
            Question: {question}
            Answer:
            """
        )
        self.load_df(df)
        self.llm = self._initialize_llm()


    def load_df(self, df):
        df.fillna({'Title': 'No Title', 'Text': 'No Text'}, inplace=True)
        self.documents = (df['Title'] + ": " + df['Text']).tolist()

        
    # def build_vector_db(self):
    #     self.embeddings = self.model.encode([text for text in self.documents])
    #     # Create a FAISS index
    #     self.index = faiss.IndexFlatL2(self.embeddings.shape[1])  # L2 distance: Euclidean distance
    #     self.index.add(np.array(self.embeddings))

    
    def search_documents(self, query, k):
        query_embedding = self.model.encode([query])
        D, I = self.index.search(np.array(query_embedding), k)
        results = [self.documents[i] for i in I[0]]
        return results if results else ["No relevant documents found"]
    

    def _initialize_llm(self):
        return HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3", 
            temperature=0.2, 
            top_k=20, 
            huggingfacehub_api_token=self.api_key  # Load token from environment variable
        )
    

    def generate_answer(self, query):
        results = self.search_documents(query, 10)
        relevant_contexts = "\n".join(results)

        inputs = {
            "context": relevant_contexts,
            "question": query
        }

        rag_chain = (
            self.prompt 
            | self.llm
            | StrOutputParser()
        )
        return rag_chain.invoke(inputs)

          

          
# Sample questions:
# What do students feel about the satisfactory/unsatisfactory option for their grades?
# What challenges do NTU university students face?
# What do NTU students discuss in this subreddit?
# What do the university students feel about the campus transport?
# What type of academic questions do the students ask in this subreddit? Be specific about the students' majors and questions asked.
# Tell me more about NTU students' daily lives.
# According to the subreddit, what are the most popular clubs in NTU?
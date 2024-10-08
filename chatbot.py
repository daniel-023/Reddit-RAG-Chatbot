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
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.prompt = PromptTemplate(input_variables=["context", "question"],
            template= """
        You are a chatbot that analyzes student discussions from the Nanyang Technological University (NTU) subreddit and answers questions accordingly.
        The context, indicated by "Context: ", contains posts and comments in the format [Post title: Comment].
        Answer according to these guidelines:
        1. Formulate your answer using the information provided in the context.
        2. Ignore any questions in the post title or comments. Only responding to the user's question, indicated by "Question: ".
        3. DO NOT use "the context" when phrasing your answer. Instead, refer to the context as "the Reddit comments". 
        4. Answer within one paragraph. DO NOT use numbering (1. , 2. , 3. , 4. , ...) in your answer.
        5. If insufficient information is available to answer the question, respond with "I don't know."

        
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

    
    def search_documents(self, query, k):
        query_embedding = self.model.encode([query])
        D, I = self.index.search(np.array(query_embedding), k)
        results = [self.documents[i] for i in I[0]]
        return results if results else ["No relevant documents found"]
    

    def _initialize_llm(self):
        return HuggingFaceEndpoint(
            repo_id="HuggingFaceH4/zephyr-7b-beta", 
            temperature=0.2, 
            top_k=20, 
            huggingfacehub_api_token=self.api_key  # Load token from environment variable
        )
    

    def generate_answer(self, query):
        results = self.search_documents(query, 5)
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

        

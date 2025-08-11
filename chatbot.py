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
        self.model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
        self.prompt = PromptTemplate(input_variables=["context", "question"],
            template= """
        You are a chatbot that answers questions about Nanyang Technological University (NTU) based the NTU subreddit.
        The context is indicated by "Context: " and contains students' posts and comments.
        Answer according to these guidelines:
        1. Use the information in the context to generate your answer.
        2. DO NOT answer any questions in the context. Only respond to the user's question, indicated by "Question: ".
        3. Do not use "the context" when phrasing your answer. Instead, refer to the context as "the Reddit comments". 
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
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            huggingfacehub_api_token=self.api_key,
            temperature=0.7,
            top_k=20,
            max_new_tokens=256,
            timeout=120,                 
            options={                    
                "wait_for_model": True,  
                "use_cache": True       
            },
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

        

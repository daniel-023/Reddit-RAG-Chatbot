import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


df = pd.read_csv('data/reddit_data.csv', parse_dates=['Timestamp'])

df.fillna({'Title': 'No Title', 'Text': 'No Text'}, inplace=True)

documents = (df['Title'] + ": " + df['Text']).tolist()

# Build vector DB
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode([text for text in documents])

# Create a FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance: Euclidean distance
index.add(np.array(embeddings))

faiss.write_index(index, 'data/faiss_index.index')

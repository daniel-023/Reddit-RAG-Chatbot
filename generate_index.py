import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


df = pd.read_csv('data/reddit_data.csv')

df.fillna({'Title': 'No Title', 'Text': 'No Text'}, inplace=True)

documents = (df['Title'] + ": " + df['Text']).tolist()

# Build vector DB
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
embeddings = model.encode([text for text in documents], show_progress_bar=True)

# Create a FAISS index
index = faiss.IndexFlatIP(embeddings.shape[1])  # IP: Inner Product
index.add(np.array(embeddings))

faiss.write_index(index, 'data/faiss_index.index')

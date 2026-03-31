import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import re
import os
import torch
from tqdm import tqdm


def chunk_text(text, max_chars=800, overlap=120):
    normalized = re.sub(r"\s+", " ", str(text)).strip()
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    chunks = []
    start = 0
    min_break = int(max_chars * 0.6)
    while start < len(normalized):
        end = min(start + max_chars, len(normalized))
        if end < len(normalized):
            break_at = normalized.rfind(" ", start + min_break, end)
            if break_at != -1:
                end = break_at
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        start = max(end - overlap, start + 1)

    return chunks


def encode_documents_in_batches(model, documents, batch_size=16):
    all_embeddings = []
    for start in tqdm(range(0, len(documents), batch_size), desc="Batches"):
        batch = documents[start:start + batch_size]
        batch_embeddings = model.encode(
            batch,
            batch_size=len(batch),
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        all_embeddings.append(batch_embeddings)

    return np.vstack(all_embeddings)


def main():
    df = pd.read_csv('data/reddit_data.csv')
    df.fillna({'Title': 'No Title', 'Text': 'No Text'}, inplace=True)

    chunk_records = []
    for _, row in df.iterrows():
        title = str(row.get("Title", "No Title"))
        text = str(row.get("Text", ""))
        combined = f"{title}: {text}".strip()
        chunks = chunk_text(combined)
        for chunk_id, chunk in enumerate(chunks):
            chunk_records.append(
                {
                    "chunk_text": chunk,
                    "title": title,
                    "timestamp": row.get("Timestamp"),
                    "post_url": row.get("Post_URL"),
                    "type": row.get("Type"),
                    "post_id": row.get("Post_id"),
                    "comment_id": row.get("Comment_id"),
                    "chunk_id": chunk_id,
                }
            )

    if not chunk_records:
        raise ValueError("No documents found in data/reddit_data.csv")

    chunk_df = pd.DataFrame(chunk_records)
    documents = chunk_df["chunk_text"].tolist()

    # Build vector DB
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    try:
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    model = SentenceTransformer('multi-qa-mpnet-base-dot-v1', device="cpu")
    embeddings = encode_documents_in_batches(model, documents, batch_size=16)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    np.save('data/embeddings.npy', embeddings)
    del model

    import faiss

    # Create a FAISS index
    index = faiss.IndexFlatIP(embeddings.shape[1])  # IP: Inner Product
    index.add(embeddings)

    faiss.write_index(index, 'data/faiss_index.index')
    chunk_df.to_csv('data/faiss_metadata.csv', index=False)


if __name__ == "__main__":
    main()

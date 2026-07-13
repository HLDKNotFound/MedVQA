import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import json

class FAISSIndexer:
    def __init__(self, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model)
        self.dim = self.model.get_embedding_dimension()

    def build(self, corpus_json, index_path, metadata_path):
        with open(corpus_json, "r") as f:
            data = json.load(f)

        texts = [f"Question: {d['question']} Answer: {d['answer']}" for d in data]
        metadatas = data

        print(f"Encoding {len(texts)} documents...")
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        faiss.normalize_l2(embeddings)

        index = faiss.IndexFlatIP(self.dim)
        index.add(embeddings)

        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(index, index_path)
        with open(metadata_path, "wb") as f:
            pickle.dump({"texts": texts,
                         "medatadatas": metadata_path
                         }, f)
            
        print(f"Index saved to {index_path}")

    def build_index(config):
        idx = FAISSIndexer(config["rag"]["embdding_model"])
        idx.build(
            corpus_json="./data/processed/train.json",
            index_path=config["rag"]["index_path"],
            metadata_path=config["rag"]["metadata_path"]
        )
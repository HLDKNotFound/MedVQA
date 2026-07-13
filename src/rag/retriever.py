import pickle
import faiss
from sentence_transformers import SentenceTransformer

class FAISSRetriever:
    def __init__(self, config):
        self.cfg = config
        self.model = SentenceTransformer(config["embedding_model"])
        self.index = faiss.read_index(config["index_path"])
        with open(config["metadata_path"], "rb") as f:
            self.store = pickle.load(f)

    def retrieve(self, query, k=None):
        k = k or self.cfg["top_k"]
        emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(emb)
        distances, indices = self.index.search(emb, k)
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.store["texts"]):
                continue

            results.append({
                "text": self.store["texts"][idx],
                "metadata": self.store["metadatas"][idx],
                "score": float(score)
            })

        return results
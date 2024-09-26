from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

# Step 1: Define the SentenceTransformerEmbeddings Class
class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str = 'distilbert-base-nli-stsb-mean-tokens'):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([text], convert_to_tensor=False).tolist()[0]
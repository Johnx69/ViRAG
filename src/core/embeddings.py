from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from ..config.settings import settings


def get_embedding_model():
    return HuggingFaceEmbedding(
        model_name=settings.embedding_model, device="cpu", embed_batch_size=256
    )


if __name__ == "__main__":
    model = get_embedding_model()

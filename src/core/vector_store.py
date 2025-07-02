import weaviate
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core import StorageContext
from ..config.settings import settings
import logging

logger = logging.getLogger(__name__)


class VectorStoreManager:
    def __init__(self):
        self.client = None
        self.vector_store = None

    def connect(self):
        try:
            # Parse URL to get host
            import re

            url_pattern = r"http://([^:]+):(\d+)"
            match = re.match(url_pattern, settings.weaviate_url)
            if not match:
                raise ValueError(
                    f"Invalid Weaviate URL format: {settings.weaviate_url}"
                )

            host = match.group(1)
            port = int(match.group(2))

            # Connect to self-hosted Weaviate
            self.client = weaviate.connect_to_custom(
                http_host=host,
                http_port=port,
                http_secure=False,
                grpc_host=host,
                grpc_port=50051,
                grpc_secure=False,
                auth_credentials=weaviate.auth.AuthApiKey(settings.weaviate_api_key),
            )
            logger.info("Connected to self-hosted Weaviate successfully")
            return self.client
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            raise

    def get_vector_store(self, collection_name: str):
        if not self.client:
            self.connect()

        self.vector_store = WeaviateVectorStore(
            weaviate_client=self.client, index_name=collection_name
        )
        return self.vector_store

    def get_storage_context(self, collection_name: str):
        vector_store = self.get_vector_store(collection_name)
        return StorageContext.from_defaults(vector_store=vector_store)

    def close(self):
        if self.client:
            self.client.close()
            logger.info("Weaviate connection closed")

    def __del__(self):
        """Ensure connection is properly closed"""
        try:
            self.close()
        except:
            pass

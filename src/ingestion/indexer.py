from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.settings import Settings
from typing import List
from ..core.embeddings import get_embedding_model
from ..core.vector_store import VectorStoreManager
from ..config.settings import settings
import logging

logger = logging.getLogger(__name__)


class DocumentIndexer:
    def __init__(self):
        self.vector_store_manager = VectorStoreManager()
        self.text_splitter = SentenceSplitter(
            chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
        )

        # Set global settings for LlamaIndex
        Settings.embed_model = get_embedding_model()

    def index_documents(
        self, documents: List, collection_name: str
    ) -> VectorStoreIndex:
        try:
            documents = documents[len(documents)//2:]
            logger.info(f"Starting indexing of {len(documents)} documents")

            # Parse documents into nodes
            nodes = self.text_splitter.get_nodes_from_documents(documents)
            logger.info(f"Created {len(nodes)} nodes from documents")

            # Get storage context
            storage_context = self.vector_store_manager.get_storage_context(
                collection_name
            )

            # Create index
            index = VectorStoreIndex(
                nodes, storage_context=storage_context, show_progress=True
            )

            logger.info(
                f"Successfully indexed documents to collection: {collection_name}"
            )
            return index

        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            raise

    def load_existing_index(self, collection_name: str) -> VectorStoreIndex:
        try:
            vector_store = self.vector_store_manager.get_vector_store(collection_name)
            index = VectorStoreIndex.from_vector_store(vector_store)
            logger.info(f"Loaded existing index from collection: {collection_name}")
            return index
        except Exception as e:
            logger.error(f"Error loading existing index: {e}")
            raise

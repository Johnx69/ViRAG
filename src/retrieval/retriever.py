from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from typing import List, Dict, Any
from ..config.settings import settings
import logging

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(self, index: VectorStoreIndex):
        self.index = index
        self.retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=settings.top_k,
            vector_store_query_mode=VectorStoreQueryMode.HYBRID,
            alpha=0.5,  # Balance between dense and sparse retrieval
        )

    def retrieve(self, query: str) -> List[Any]:
        try:
            logger.info(f"Retrieving documents for query: {query[:100]}...")
            nodes = self.retriever.retrieve(query)
            logger.info(f"Retrieved {len(nodes)} nodes")
            return nodes
        except Exception as e:
            logger.error(f"Error in retrieval: {e}")
            return []

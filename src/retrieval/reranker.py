from llama_index.core.postprocessor import SentenceTransformerRerank
from typing import List, Any
from ..config.settings import settings
import logging

logger = logging.getLogger(__name__)


class DocumentReranker:
    def __init__(self, top_n: int = None):
        self.top_n = top_n or settings.top_k
        self.reranker = SentenceTransformerRerank(
            top_n=self.top_n, model="BAAI/bge-reranker-v2-m3", device="cpu"
        )

    def rerank(self, nodes: List[Any], query: str) -> tuple[List[Any], float]:
        try:
            logger.info(f"Reranking {len(nodes)} nodes")
            reranked_nodes = self.reranker.postprocess_nodes(nodes, query_str=query)

            # Calculate confidence score based on top result
            confidence_score = 0.0
            if reranked_nodes:
                # Use the score from the top reranked node
                confidence_score = getattr(reranked_nodes[0], "score", 0.0)

            logger.info(
                f"Reranked to {len(reranked_nodes)} nodes, confidence: {confidence_score}"
            )
            return reranked_nodes, confidence_score

        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            return nodes, 0.0

    def meets_confidence_threshold(self, confidence_score: float) -> bool:
        return confidence_score >= settings.confidence_threshold

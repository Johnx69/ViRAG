from typing import Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..router.query_router import QueryRouter
from ..retrieval.retriever import HybridRetriever
from ..retrieval.reranker import DocumentReranker
from ..search.web_search import WebSearcher
from ..sql_agent import SQLAgent  # New import
from ..core.llm import get_llm
from ..ingestion.indexer import DocumentIndexer
from ..config.settings import settings
import logging
from ..prompts import RAG_RESPONSE

logger = logging.getLogger(__name__)


class VietnameseRAGPipeline:
    def __init__(self):
        self.router = QueryRouter()
        self.indexer = DocumentIndexer()
        self.reranker = DocumentReranker()
        self.web_searcher = WebSearcher()
        self.llm = get_llm()

        # Initialize SQL Agent
        db_path = settings.ROOT / "data" / "db" / "sell_data.sqlite"
        self.sql_agent = SQLAgent(str(db_path), "plots")

        # Load or create indexes
        self.wikipedia_index = None
        self._load_indexes()

        # Create response generation chain
        self.response_chain = self._create_response_chain()

    def _load_indexes(self):
        """Load existing indexes or prepare for indexing."""
        try:
            self.wikipedia_index = self.indexer.load_existing_index(
                settings.wikipedia_collection
            )
            logger.info("Loaded existing Wikipedia index")
        except Exception as e:
            logger.warning(f"Could not load existing index: {e}")
            logger.info("Wikipedia index will need to be created")

    def _create_response_chain(self):
        """Create the response generation chain."""
        prompt_template = RAG_RESPONSE

        prompt = ChatPromptTemplate.from_template(prompt_template)
        return prompt | self.llm | StrOutputParser()

    def index_wikipedia_data(self, num_samples: Optional[int] = None):
        """Index Vietnamese Wikipedia data."""
        from ..ingestion.data_loader import VietnameseWikipediaLoader

        try:
            loader = VietnameseWikipediaLoader()
            documents = loader.load_data(num_samples)

            self.wikipedia_index = self.indexer.index_documents(
                documents, settings.wikipedia_collection
            )
            logger.info("Successfully indexed Wikipedia data")

        except Exception as e:
            logger.error(f"Error indexing Wikipedia data: {e}")
            raise

    def query(self, question: str) -> Dict[str, Any]:
        """Main query processing method."""
        try:
            logger.info(f"Processing query: {question}")

            # Route the question
            route_result = self.router.route(question)

            if route_result.datasource == "selling_database":
                return self._handle_selling_query(question)
            else:
                return self._handle_general_query(question)

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "answer": "Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi của bạn.",
                "source": "error",
                "error": str(e),
            }

    def _handle_selling_query(self, question: str) -> Dict[str, Any]:
        """Handle selling/business related queries using SQL agent."""
        try:
            logger.info("Routing to SQL agent")
            result = self.sql_agent.query(question)

            # Add plot information to response if available
            if result.get("plot_paths"):
                plot_info = f"\n\nĐã tạo {len(result['plot_paths'])} biểu đồ minh họa."
                result["answer"] += plot_info

            return result

        except Exception as e:
            logger.error(f"Error in SQL agent: {e}")
            return {
                "answer": "Xin lỗi, đã xảy ra lỗi khi truy vấn cơ sở dữ liệu bán hàng.",
                "source": "selling_database",
                "error": str(e),
            }

    def _handle_general_query(self, question: str) -> Dict[str, Any]:
        """Handle general knowledge queries using corrective/adaptive RAG."""
        if not self.wikipedia_index:
            return {
                "answer": "Cơ sở dữ liệu Wikipedia chưa được tạo. Vui lòng chạy indexing trước.",
                "source": "error",
            }

        # Step 1: Retrieve from Wikipedia
        retriever = HybridRetriever(self.wikipedia_index)
        retrieved_nodes = retriever.retrieve(question)

        if not retrieved_nodes:
            return self._fallback_to_web_search(question)

        # Step 2: Rerank documents
        reranked_nodes, confidence_score = self.reranker.rerank(
            retrieved_nodes, question
        )

        # Step 3: Check confidence threshold
        if self.reranker.meets_confidence_threshold(confidence_score):
            # High confidence - use retrieved documents
            context = self._format_nodes_for_context(reranked_nodes)
            answer = self.response_chain.invoke(
                {"context": context, "question": question}
            )

            return {
                "answer": answer,
                "source": "wikipedia",
                "confidence_score": confidence_score,
                "num_sources": len(reranked_nodes),
            }
        else:
            # Low confidence - fallback to web search
            return self._fallback_to_web_search(question)

    def _fallback_to_web_search(self, question: str) -> Dict[str, Any]:
        """Fallback to web search when confidence is low."""
        logger.info("Falling back to web search due to low confidence")

        # Perform web search
        web_results = self.web_searcher.search(question)

        if not web_results:
            return {
                "answer": "Xin lỗi, tôi không thể tìm thấy thông tin để trả lời câu hỏi của bạn.",
                "source": "no_results",
            }

        # Format web results as context
        context = self.web_searcher.format_search_results(web_results)

        # Generate answer using web search results
        answer = self.response_chain.invoke({"context": context, "question": question})

        return {
            "answer": answer,
            "source": "web_search",
            "num_sources": len(web_results),
        }

    def _format_nodes_for_context(self, nodes) -> str:
        """Format retrieved nodes for LLM context."""
        if not nodes:
            return "Không có thông tin liên quan."

        context_parts = []
        for i, node in enumerate(nodes, 1):
            title = node.metadata.get("title", "Unknown")
            content = node.text[:1000] + "..." if len(node.text) > 1000 else node.text
            context_parts.append(f"Nguồn {i} ({title}):\n{content}")

        return "\n\n".join(context_parts)

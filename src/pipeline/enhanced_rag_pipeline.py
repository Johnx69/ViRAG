from typing import Dict, Any, Optional, List, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..router.query_router import QueryRouter
from ..retrieval.retriever import HybridRetriever
from ..retrieval.reranker import DocumentReranker
from ..retrieval.query_rewriter import QueryRewriter
from ..search.web_search import WebSearcher
from ..sql_agent import SQLAgent
from ..core.llm import get_llm
from ..ingestion.indexer import DocumentIndexer
from ..config.settings import settings
import logging

logger = logging.getLogger(__name__)


class EnhancedVietnameseRAGPipeline:
    """Enhanced RAG Pipeline with query rewriting capabilities."""

    def __init__(self):
        self.router = QueryRouter()
        self.indexer = DocumentIndexer()
        self.reranker = DocumentReranker()
        self.web_searcher = WebSearcher()
        self.query_rewriter = QueryRewriter()
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
        from ..prompts.rag_response import RAG_RESPONSE

        prompt = ChatPromptTemplate.from_template(RAG_RESPONSE)
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

    def query(
        self,
        question: str,
        rewrite_strategies: Optional[List[str]] = None,
        max_web_searches: int = 3,
        max_return_docs: int = 5,
    ) -> Dict[str, Any]:
        """
        Main query processing method with enhanced rewriting capabilities.

        Args:
            question: User question
            rewrite_strategies: List of strategies to apply ['simple', 'decompose', 'hyde']
            max_web_searches: Maximum number of web searches to perform
            max_return_docs: Maximum number of documents to return
        """
        try:
            logger.info(f"Processing query: {question}")

            # Route the question
            route_result = self.router.route(question)

            if route_result.datasource == "selling_database":
                return self._handle_selling_query(question)
            else:
                return self._handle_general_query_with_rewriting(
                    question, rewrite_strategies, max_web_searches, max_return_docs
                )

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "answer": "Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi của bạn.",
                "source": "error",
                "error": str(e),
                "process_log": [f"Error: {str(e)}"],
                "retrieved_docs": [],
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

            # Add process log
            result["process_log"] = [
                "Định tuyến đến cơ sở dữ liệu bán hàng",
                f"Thực thi SQL: {result.get('sql_query', 'N/A')}",
                "Tạo câu trả lời từ kết quả SQL",
            ]

            if result.get("plot_paths"):
                result["process_log"].append(f"Tạo {len(result['plot_paths'])} biểu đồ")

            return result

        except Exception as e:
            logger.error(f"Error in SQL agent: {e}")
            return {
                "answer": "Xin lỗi, đã xảy ra lỗi khi truy vấn cơ sở dữ liệu bán hàng.",
                "source": "selling_database",
                "error": str(e),
                "process_log": [f"Lỗi SQL agent: {str(e)}"],
                "retrieved_docs": [],
            }

    def _handle_general_query_with_rewriting(
        self,
        question: str,
        rewrite_strategies: Optional[List[str]],
        max_web_searches: int,
        max_return_docs: int,
    ) -> Dict[str, Any]:
        """Handle general knowledge queries with enhanced rewriting."""
        process_log = []

        if not self.wikipedia_index:
            return {
                "answer": "Cơ sở dữ liệu Wikipedia chưa được tạo. Vui lòng chạy indexing trước.",
                "source": "error",
                "process_log": ["Lỗi: Chưa có index Wikipedia"],
                "retrieved_docs": [],
            }

        # Step 1: Try initial retrieval
        process_log.append("Bước 1: Tìm kiếm ban đầu trong Wikipedia")
        retriever = HybridRetriever(self.wikipedia_index)
        best_result = self._try_retrieval_and_rerank(question, retriever, process_log)

        if best_result["success"]:
            best_result["process_log"] = process_log
            return best_result

        # Step 2: Apply rewriting strategies if provided
        if rewrite_strategies:
            process_log.append(
                f"Bước 2: Áp dụng các chiến lược viết lại: {rewrite_strategies}"
            )

            rewrite_results = self.query_rewriter.apply_strategies(
                question, rewrite_strategies
            )

            for i, rewrite_result in enumerate(rewrite_results):
                strategy = rewrite_result["strategy"]
                process_log.append(f"Thử chiến lược {strategy} (lần {i+1})")

                if strategy == "decompose":
                    # For decompose, try each sub-question
                    for j, sub_query in enumerate(rewrite_result["queries"]):
                        process_log.append(f"  Câu hỏi con {j+1}: {sub_query}")
                        result = self._try_retrieval_and_rerank(
                            sub_query, retriever, process_log
                        )
                        if result["success"]:
                            result["process_log"] = process_log
                            result["rewrite_info"] = {
                                "strategy": strategy,
                                "original_query": question,
                                "rewritten_query": sub_query,
                                "attempt": i + 1,
                            }
                            return result
                else:
                    # For simple and HyDE, try the rewritten query
                    rewritten_query = rewrite_result["queries"][0]
                    process_log.append(f"  Câu hỏi đã viết lại: {rewritten_query}")
                    result = self._try_retrieval_and_rerank(
                        rewritten_query, retriever, process_log
                    )
                    if result["success"]:
                        result["process_log"] = process_log
                        result["rewrite_info"] = {
                            "strategy": strategy,
                            "original_query": question,
                            "rewritten_query": rewritten_query,
                            "attempt": i + 1,
                        }
                        return result

        # Step 3: Fallback to web search
        process_log.append("Bước 3: Chuyển sang tìm kiếm web")
        return self._fallback_to_web_search(question, max_web_searches, process_log)

    def _try_retrieval_and_rerank(
        self, query: str, retriever: HybridRetriever, process_log: List[str]
    ) -> Dict[str, Any]:
        """Try retrieval and reranking for a given query."""
        try:
            # Retrieve documents
            retrieved_nodes = retriever.retrieve(query)
            process_log.append(f"  Tìm thấy {len(retrieved_nodes)} tài liệu")

            if not retrieved_nodes:
                process_log.append("  Không tìm thấy tài liệu nào")
                return {"success": False}

            # Rerank documents
            reranked_nodes, confidence_score = self.reranker.rerank(
                retrieved_nodes, query
            )
            process_log.append(
                f"  Điểm tin cậy sau khi xếp hạng: {confidence_score:.3f}"
            )

            # Check confidence threshold
            if self.reranker.meets_confidence_threshold(confidence_score):
                process_log.append("  Đạt ngưỡng tin cậy - Tạo câu trả lời")

                # Format retrieved documents for response
                retrieved_docs = []
                for i, node in enumerate(reranked_nodes):
                    doc_info = {
                        "title": node.metadata.get("title", f"Document {i+1}"),
                        "content": (
                            node.text[:200] + "..."
                            if len(node.text) > 200
                            else node.text
                        ),
                        "score": getattr(node, "score", 0.0),
                        "url": node.metadata.get("url", ""),
                    }
                    retrieved_docs.append(doc_info)

                # Generate answer
                context = self._format_nodes_for_context(reranked_nodes)
                answer = self.response_chain.invoke(
                    {"context": context, "question": query}
                )

                return {
                    "success": True,
                    "answer": answer,
                    "source": "wikipedia",
                    "confidence_score": confidence_score,
                    "num_sources": len(reranked_nodes),
                    "retrieved_docs": retrieved_docs,
                }
            else:
                process_log.append("  Không đạt ngưỡng tin cậy")
                return {"success": False}

        except Exception as e:
            process_log.append(f"  Lỗi khi tìm kiếm: {str(e)}")
            logger.error(f"Error in retrieval: {e}")
            return {"success": False}

    def _fallback_to_web_search(
        self, question: str, max_searches: int, process_log: List[str]
    ) -> Dict[str, Any]:
        """Fallback to web search when confidence is low."""
        logger.info("Falling back to web search due to low confidence")
        process_log.append("Sử dụng tìm kiếm web do không tìm thấy thông tin phù hợp")

        # Perform web search
        web_results = self.web_searcher.search(question)
        process_log.append(f"Tìm thấy {len(web_results)} kết quả web")

        if not web_results:
            return {
                "answer": "Xin lỗi, tôi không thể tìm thấy thông tin để trả lời câu hỏi của bạn.",
                "source": "no_results",
                "process_log": process_log,
                "retrieved_docs": [],
            }

        # Limit web results
        web_results = web_results[:max_searches]

        # Format web results as context
        context = self.web_searcher.format_search_results(web_results)

        # Generate answer using web search results
        answer = self.response_chain.invoke({"context": context, "question": question})

        # Format web results for UI
        retrieved_docs = []
        for i, doc in enumerate(web_results):
            doc_info = {
                "title": doc.metadata.get("title", f"Web Result {i+1}"),
                "content": (
                    doc.page_content[:200] + "..."
                    if len(doc.page_content) > 200
                    else doc.page_content
                ),
                "score": doc.metadata.get("score", 0.0),
                "url": doc.metadata.get("url", ""),
            }
            retrieved_docs.append(doc_info)

        process_log.append("Tạo câu trả lời từ kết quả tìm kiếm web")

        return {
            "answer": answer,
            "source": "web_search",
            "num_sources": len(web_results),
            "process_log": process_log,
            "retrieved_docs": retrieved_docs,
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

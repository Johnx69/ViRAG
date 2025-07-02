# src/retrieval/query_rewriter.py

from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from ..config.settings import settings
import logging

logger = logging.getLogger(__name__)


class QueryRewriter:
    """Handles different query rewriting strategies for improved retrieval."""

    def __init__(self):
        # Use GPT-4o for query rewriting
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            api_key=settings.openai_api_key,  # Add this to settings
        )

        self.simple_rewriter = self._create_simple_rewriter()
        self.decompose_rewriter = self._create_decompose_rewriter()
        self.hyde_rewriter = self._create_hyde_rewriter()

    def _create_simple_rewriter(self):
        """Create simple query rewriting chain."""
        prompt = ChatPromptTemplate.from_template(
            """
        Bạn là một chuyên gia viết lại câu hỏi để cải thiện tìm kiếm thông tin.
        
        Nhiệm vụ: Viết lại câu hỏi sau để tối ưu hóa việc tìm kiếm trong cơ sở dữ liệu vector.
        Hãy làm cho câu hỏi rõ ràng hơn, cụ thể hơn và dễ tìm kiếm hơn.
        
        Câu hỏi gốc: {original_query}
        
        Hướng dẫn:
        - Mở rộng các từ viết tắt
        - Thêm các từ khóa liên quan
        - Làm rõ ý nghĩa của câu hỏi
        - Giữ nguyên ngôn ngữ tiếng Việt
        - Chỉ trả về câu hỏi đã được viết lại, không giải thích
        
        Câu hỏi đã viết lại:
        """
        )

        return prompt | self.llm | StrOutputParser()

    def _create_decompose_rewriter(self):
        """Create query decomposition chain."""
        prompt = ChatPromptTemplate.from_template(
            """
        Bạn là một chuyên gia phân tích câu hỏi phức tạp thành các câu hỏi con đơn giản hơn.
        
        Nhiệm vụ: Phân tích câu hỏi phức tạp sau thành 2-3 câu hỏi con cụ thể và dễ tìm kiếm hơn.
        
        Câu hỏi gốc: {original_query}
        
        Hướng dẫn:
        - Chia thành các câu hỏi con độc lập
        - Mỗi câu hỏi con tập trung vào một khía cạnh cụ thể
        - Sử dụng ngôn ngữ tiếng Việt
        - Trả về danh sách các câu hỏi, mỗi câu hỏi trên một dòng
        - Bắt đầu mỗi câu hỏi bằng dấu "- "
        
        Các câu hỏi con:
        """
        )

        return prompt | self.llm | StrOutputParser()

    def _create_hyde_rewriter(self):
        """Create HyDE (Hypothetical Document Embeddings) chain."""
        prompt = ChatPromptTemplate.from_template(
            """
        Bạn là một chuyên gia tạo ra tài liệu giả định để cải thiện tìm kiếm.
        
        Nhiệm vụ: Tạo ra một đoạn văn bản giả định (hypothetical document) mà có thể chứa câu trả lời cho câu hỏi sau. Đoạn văn bản này sẽ được sử dụng để tìm kiếm các tài liệu liên quan.
        
        Câu hỏi: {original_query}
        
        Hướng dẫn:
        - Viết một đoạn văn bản 2-3 câu
        - Giả định như đây là một phần của tài liệu Wikipedia tiếng Việt
        - Sử dụng ngôn ngữ tự nhiên và thuật ngữ chuyên môn phù hợp
        - Bao gồm các từ khóa và khái niệm liên quan
        - Không nói rằng đây là giả định hay không chắc chắn
        
        Tài liệu giả định:
        """
        )

        return prompt | self.llm | StrOutputParser()

    def rewrite_simple(self, query: str) -> str:
        """Simple query rewriting."""
        try:
            rewritten = self.simple_rewriter.invoke({"original_query": query})
            logger.info(f"Simple rewrite: '{query}' -> '{rewritten}'")
            return rewritten.strip()
        except Exception as e:
            logger.error(f"Error in simple rewriting: {e}")
            return query

    def rewrite_decompose(self, query: str) -> List[str]:
        """Decompose query into sub-questions."""
        try:
            response = self.decompose_rewriter.invoke({"original_query": query})

            # Parse the response into a list of questions
            questions = []
            for line in response.split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    questions.append(line[2:].strip())
                elif line and not line.startswith("Các câu hỏi con:"):
                    questions.append(line.strip())

            if not questions:
                questions = [query]  # Fallback to original query

            logger.info(f"Decomposed '{query}' into {len(questions)} sub-questions")
            return questions
        except Exception as e:
            logger.error(f"Error in decomposition: {e}")
            return [query]

    def rewrite_hyde(self, query: str) -> str:
        """Generate hypothetical document for HyDE technique."""
        try:
            hypothetical_doc = self.hyde_rewriter.invoke({"original_query": query})
            logger.info(f"Generated HyDE document for: '{query}'")
            return hypothetical_doc.strip()
        except Exception as e:
            logger.error(f"Error in HyDE generation: {e}")
            return query

    def apply_strategies(
        self, query: str, strategies: List[str]
    ) -> List[Dict[str, Any]]:
        """Apply multiple rewriting strategies in sequence."""
        results = []

        for strategy in strategies:
            if strategy == "simple":
                rewritten = self.rewrite_simple(query)
                results.append(
                    {
                        "strategy": "simple",
                        "original": query,
                        "rewritten": rewritten,
                        "queries": [rewritten],
                    }
                )
                # Use the rewritten query for next strategy
                query = rewritten

            elif strategy == "decompose":
                sub_questions = self.rewrite_decompose(query)
                results.append(
                    {
                        "strategy": "decompose",
                        "original": query,
                        "rewritten": query,
                        "queries": sub_questions,
                    }
                )
                # For next strategy, use the first sub-question or combined
                if sub_questions:
                    query = sub_questions[0]

            elif strategy == "hyde":
                hypothetical = self.rewrite_hyde(query)
                results.append(
                    {
                        "strategy": "hyde",
                        "original": query,
                        "rewritten": hypothetical,
                        "queries": [hypothetical],
                    }
                )
                # Use hypothetical document for next strategy
                query = hypothetical

        return results

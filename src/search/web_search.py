# First install the new package: pip install langchain-tavily
from langchain_tavily import TavilySearch
from langchain.schema import Document
from typing import List
from ..config.settings import settings
import logging

logger = logging.getLogger(__name__)


class WebSearcher:
    def __init__(self):
        # Use max_results instead of k for the new API
        self.search_tool = TavilySearch(
            max_results=5, 
            api_key=settings.tavily_api_key
        )

    def search(self, query: str) -> List[Document]:
        try:
            logger.info(f"Performing web search for: {query}")
            response = self.search_tool.invoke({"query": query})
            
            # Debug: print the response structure
            logger.debug(f"Tavily response: {response}")
            
            # Extract results from the response
            if isinstance(response, dict) and 'results' in response:
                results = response['results']
            elif isinstance(response, list):
                # Fallback if it's a direct list
                results = response
            else:
                logger.error(f"Unexpected response format: {type(response)}")
                return []

            documents = []
            for result in results:
                # Handle both dict and object formats
                if isinstance(result, dict):
                    content = result.get("content", "")
                    title = result.get("title", "")
                    url = result.get("url", "")
                    score = result.get("score", 0.0)
                else:
                    # If result is an object with attributes
                    content = getattr(result, "content", "")
                    title = getattr(result, "title", "")
                    url = getattr(result, "url", "")
                    score = getattr(result, "score", 0.0)
                
                doc = Document(
                    page_content=content,
                    metadata={
                        "title": title,
                        "url": url,
                        "source": "web_search",
                        "score": score,
                    },
                )
                documents.append(doc)

            logger.info(f"Found {len(documents)} web search results")
            return documents

        except Exception as e:
            logger.error(f"Error in web search: {e}")
            return []

    def format_search_results(self, documents: List[Document]) -> str:
        """Format search results for LLM context."""
        if not documents:
            return "No web search results found."

        formatted_results = []
        for i, doc in enumerate(documents, 1):
            title = doc.metadata.get("title", "Unknown Title")
            url = doc.metadata.get("url", "")
            content = (
                doc.page_content[:500] + "..."
                if len(doc.page_content) > 500
                else doc.page_content
            )

            formatted_results.append(
                f"Result {i}:\n"
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Content: {content}\n"
            )

        return "\n".join(formatted_results)
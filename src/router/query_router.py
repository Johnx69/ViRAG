from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from ..core.llm import get_llm
import logging
from ..prompts import ROUTER_SYSTEM, ROUTER_HUMAN

logger = logging.getLogger(__name__)


class RouteQuery(BaseModel):
    """Route a user query to the appropriate data source."""

    datasource: Literal["selling_database", "general_knowledge"] = Field(
        ...,
        description="Route to selling_database for sales-related queries or general_knowledge for other questions",
    )
    reasoning: str = Field(..., description="Brief explanation of the routing decision")


class QueryRouter:
    def __init__(self):
        self.llm = get_llm()
        self.structured_llm = self.llm.with_structured_output(RouteQuery)
        self.prompt = self._create_prompt()
        self.router_chain = self.prompt | self.structured_llm

    def _create_prompt(self):

        return ChatPromptTemplate.from_messages(
            [
                ("system", ROUTER_SYSTEM),
                ("human", ROUTER_HUMAN),
            ]
        )

    def route(self, question: str) -> RouteQuery:
        try:
            result = self.router_chain.invoke({"question": question})
            logger.info(f"Routed question to: {result.datasource} - {result.reasoning}")
            return result
        except Exception as e:
            logger.error(f"Error in routing: {e}")
            # Default to general knowledge if routing fails
            return RouteQuery(
                datasource="general_knowledge",
                reasoning="Fallback due to routing error",
            )

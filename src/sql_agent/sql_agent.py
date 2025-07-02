# src/sql_agent/sql_agent.py

from typing import Dict, Any, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from ..core.llm import get_llm
from .sql_tools import SQLToolkit
from .plotting_agent import PlottingAgent
import logging
import json
from ..prompts import SQL_SCHEMA, SQL_GENERATION, SELLING_FORMAT, SQL_FIX

logger = logging.getLogger(__name__)


class SQLAgentState(dict):
    """State for the SQL agent."""

    question: str
    sql_query: str
    query_result: str
    formatted_answer: str
    plot_needed: bool
    plot_paths: List[str]
    error_count: int
    messages: List[Any]
    schema_info: str


class SQLAgent:
    """Advanced SQL agent with comprehensive database schema knowledge."""

    def __init__(self, db_path: str, plot_output_dir: str = "plots"):
        self.llm = get_llm()
        self.db_path = db_path
        self.sql_toolkit = SQLToolkit(db_path, self.llm)
        self.plotting_agent = PlottingAgent(plot_output_dir)
        self.tools = self.sql_toolkit.get_tools()
        self.tool_node = ToolNode(self.tools)

        # Comprehensive database schema information
        self.database_schema = self._get_comprehensive_schema()

        # Create the agent graph
        self.graph = self._create_graph()

    def _get_comprehensive_schema(self) -> str:
        """Get comprehensive database schema information."""
        return SQL_SCHEMA

    def _create_graph(self):
        """Create the agent execution graph."""

        def analyze_question(state: SQLAgentState) -> SQLAgentState:
            """Analyze the question and determine requirements."""

            # Determine if plotting is needed
            # Keywords & phrases that strongly imply the user wants a CHART / PLOT
            # Vietnamese + English, lower-case, ASCII-only where possible.
            plot_keywords = [
                # ── Generic verbs & nouns ───────────────────────────────────────────
                "plot",
                "draw",
                "diagram",
                "figure",
                "graph",
                "chart",
                "visualize",
                "visualization",
                "viz",
                "render",
                "illustrate",
                "depict",
                "show trend",
                "data viz",
                "minh hoạ",
                "minh họa",
                "hình vẽ",
                "biểu diễn",
                "vẽ biểu đồ",
                "trực quan",
                "trực quan hóa",
                "trực quan hóa",
                # ── Explicit Vietnamese requests for a chart ───────────────────────
                "biểu đồ",
                "đồ thị",
                "đồ hoạ",
                "đồ họa",
                "sơ đồ",
                "hình minh hoạ",
                "hình minh họa",
                "đồ thị cột",
                "đồ thị đường",
                "đồ thị tròn",
                "vẽ đồ thị",
                "lập biểu đồ",
                "minh họa dữ liệu",
                "hình trực quan",
                # ── Time-series & trend cues ───────────────────────────────────────
                "trend",
                "xu hướng",
                "xu huong",
                "theo thời gian",
                "timeline",
                "over time",
                "time series",
                "temporal",
                "diễn biến",
                "diễn tiến",
                "evolution",
                "daily pattern",
                "monthly pattern",
                "growth curve",
                # ── Comparative / distribution prompts ─────────────────────────────
                "compare",
                "comparison",
                "phân bố",
                "phân phối",
                "distribution",
                "histogram",
                "density",
                "boxplot",
                "box plot",
                "violin",
                "whisker",
                "sự phân bổ",
                "so sánh",
                "so sanh",
                "chiếm tỷ lệ",
                "proportion",
                "cumulative",
                "cumulative sum",
                "cumulative distribution",
                # ── Common chart types (English) ───────────────────────────────────
                #   Include both singular & plural where spelling differs.
                "bar",
                "bar chart",
                "column chart",
                "stacked bar",
                "100% stacked",
                "line",
                "line chart",
                "area chart",
                "stacked area",
                "step chart",
                "pie",
                "pie chart",
                "donut",
                "donut chart",
                "sunburst",
                "treemap",
                "scatter",
                "scatterplot",
                "bubble chart",
                "heatmap",
                "heat map",
                "candlestick",
                "ohlc",
                "waterfall chart",
                "sankey",
                "funnel chart",
                "gantt",
                "polar chart",
                "radar",
                "spider chart",
                "choropleth",
                "geo map",
                "map plot",
                "hexbin",
                "violin plot",
                "density plot",
                # ── Common chart types (Vietnamese) ────────────────────────────────
                "biểu đồ cột",
                "biểu đồ hàng",
                "biểu đồ tròn",
                "biểu đồ đường",
                "biểu đồ miền",
                "biểu đồ diện tích",
                "biểu đồ phân tán",
                "biểu đồ bong bóng",
                "biểu đồ nhiệt",
                "bản đồ nhiệt",
                "biểu đồ hộp",
                "biểu đồ violin",
                "biểu đồ thác nước",
                "biểu đồ radar",
                "biểu đồ mạng nhện",
                "biểu đồ gantt",
                "biểu đồ sunburst",
                "biểu đồ treemap",
                "biểu đồ funnel",
                # ── Business / KPI cues that almost always need visuals ────────────
                "doanh thu",
                "revenue",
                "turnover",
                "profit trend",
                "lợi nhuận",
                "kpi",
                "traffic trend",
                "conversion rate",
                "tỷ lệ chuyển đổi",
                "market share",
                "thị phần",
                "increase vs decrease",
                "sales pattern",
                # ── Statistical / analytical words implying visual summary ─────────
                "correlation",
                "tương quan",
                "regression line",
                "pca plot",
                "scatter matrix",
                "pairplot",
                "roc curve",
                "confusion matrix",
                "heat map",
                "ma plot",
                "volcano plot",
                "qq plot",
                "kaplan-meier",
                "survival curve",
                "anova plot",
                "manhattan plot",
                # ── Questions that hint at a need for shape, pattern or seasonality ─
                "mô hình theo mùa",
                "seasonality",
                "seasonal pattern",
                "chu kỳ",
                "cycle",
                "tính chu kỳ",
                "pattern recognition",
                "phân cụm",
                "cluster plot",
                "elbow plot",
                "silhouette",
                # ── Generic “show me” phrases ──────────────────────────────────────
                "cho tôi xem biểu đồ",
                "hiển thị đồ thị",
                "làm đồ thị",
                "display chart",
                "show chart",
                "plot it",
                "plot this",
                "make a graph",
            ]

            needs_plot = any(
                keyword in state["question"].lower() for keyword in plot_keywords
            )

            state["plot_needed"] = needs_plot
            state["error_count"] = 0
            state["messages"] = [HumanMessage(content=state["question"])]
            state["schema_info"] = self.database_schema

            logger.info(f"Question analyzed. Plot needed: {needs_plot}")
            return state

        def generate_sql_with_schema(state: SQLAgentState) -> SQLAgentState:
            """Generate SQL query with comprehensive schema knowledge."""

            # Create comprehensive prompt with schema
            sql_generation_prompt = SQL_GENERATION.format(
                database_schema=self.database_schema, question=state["question"]
            )

            try:
                response = self.llm.invoke(sql_generation_prompt)
                sql_query = self._extract_sql_from_response(response.content)
                state["sql_query"] = sql_query
                logger.info(f"Generated SQL: {sql_query}")

            except Exception as e:
                logger.error(f"Error generating SQL: {e}")
                state["sql_query"] = ""

            return state

        def execute_sql_with_retry(state: SQLAgentState) -> SQLAgentState:
            """Execute SQL with comprehensive error handling and retry."""

            if not state.get("sql_query"):
                state["query_result"] = "Error: No SQL query generated"
                return state

            query_tool = next(
                tool for tool in self.tools if tool.name == "sql_db_query"
            )

            try:
                result = query_tool._run(state["sql_query"])

                if "error" in result.lower():
                    state["error_count"] = state.get("error_count", 0) + 1
                    logger.warning(
                        f"SQL Error (attempt {state['error_count']}): {result}"
                    )

                    if state["error_count"] < 3:
                        # Try to fix the query
                        return self._fix_sql_with_detailed_schema(state, result)
                    else:
                        state["query_result"] = f"Failed after 3 attempts: {result}"
                else:
                    state["query_result"] = result
                    logger.info("SQL executed successfully")

            except Exception as e:
                logger.error(f"Execution error: {e}")
                state["query_result"] = f"Execution error: {str(e)}"

            return state

        def format_vietnamese_answer(state: SQLAgentState) -> SQLAgentState:
            """Format answer in Vietnamese with context."""

            if (
                not state.get("query_result")
                or "error" in state.get("query_result", "").lower()
            ):
                state["formatted_answer"] = (
                    "Xin lỗi, tôi không thể truy vấn được dữ liệu cần thiết để trả lời câu hỏi của bạn."
                )
                return state

            format_prompt = SELLING_FORMAT.format(
                question=state["question"],
                sql_query=state["sql_query"],
                query_result=state["query_result"],
            )

            try:
                response = self.llm.invoke(format_prompt)
                state["formatted_answer"] = response.content

            except Exception as e:
                logger.error(f"Error formatting answer: {e}")
                state["formatted_answer"] = (
                    f"Có lỗi khi định dạng câu trả lời: {str(e)}"
                )

            return state

        def create_plots(state: SQLAgentState) -> SQLAgentState:
            """Create plots if needed and data is available."""

            if (
                state.get("plot_needed")
                and state.get("query_result")
                and "error" not in state.get("query_result", "").lower()
            ):

                try:
                    plots = self.plotting_agent.create_plots(
                        question=state["question"],
                        sql_query=state["sql_query"],
                        query_result=state["query_result"],
                    )
                    state["plot_paths"] = plots
                    logger.info(f"Created {len(plots)} plots")

                except Exception as e:
                    logger.error(f"Error creating plots: {e}")
                    state["plot_paths"] = []
            else:
                state["plot_paths"] = []

            return state

        def should_retry_sql(state: SQLAgentState) -> str:
            """Determine if SQL should be retried."""
            if (
                state.get("query_result")
                and "error" not in state["query_result"].lower()
            ):
                return "format_answer"
            elif state.get("error_count", 0) < 3:
                return "generate_sql"
            else:
                return "format_answer"

        def should_create_plots(state: SQLAgentState) -> str:
            """Determine if plots should be created."""
            if (
                state.get("plot_needed")
                and state.get("query_result")
                and "error" not in state.get("query_result", "").lower()
            ):
                return "create_plots"
            return "end"

        # Build the graph
        workflow = StateGraph(SQLAgentState)

        # Add nodes
        workflow.add_node("analyze_question", analyze_question)
        workflow.add_node("generate_sql", generate_sql_with_schema)
        workflow.add_node("execute_sql", execute_sql_with_retry)
        workflow.add_node("format_answer", format_vietnamese_answer)
        workflow.add_node("create_plots", create_plots)

        # Add edges
        workflow.add_edge(START, "analyze_question")
        workflow.add_edge("analyze_question", "generate_sql")
        workflow.add_edge("generate_sql", "execute_sql")

        # Conditional edges
        workflow.add_conditional_edges(
            "execute_sql",
            should_retry_sql,
            {
                "format_answer": "format_answer",
                "generate_sql": "generate_sql",
            },
        )

        workflow.add_conditional_edges(
            "format_answer",
            should_create_plots,
            {"create_plots": "create_plots", "end": END},
        )

        workflow.add_edge("create_plots", END)

        return workflow.compile()

    def _extract_sql_from_response(self, response: str) -> str:
        """Extract SQL query from LLM response."""
        response = response.strip()

        # Remove markdown formatting
        if "```sql" in response:
            start = response.find("```sql") + 6
            end = response.find("```", start)
            if end != -1:
                response = response[start:end]
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end != -1:
                response = response[start:end]

        # Clean up the query
        response = response.strip()
        if response.endswith(";"):
            response = response[:-1]

        return response

    def _fix_sql_with_detailed_schema(
        self, state: SQLAgentState, error_message: str
    ) -> SQLAgentState:
        """Fix SQL query using detailed schema knowledge."""

        fix_prompt = SQL_FIX.format(
            sql_schema=self.database_schema,
            sql_query=state["sql_query"],
            error_message=error_message,
            question=state["question"],
        )

        try:
            response = self.llm.invoke(fix_prompt)
            corrected_query = self._extract_sql_from_response(response.content)
            state["sql_query"] = corrected_query
            logger.info(
                f"Corrected SQL (attempt {state['error_count']}): {corrected_query}"
            )

        except Exception as e:
            logger.error(f"Error fixing SQL: {e}")

        return state

    def query(self, question: str) -> Dict[str, Any]:
        """Process a question and return comprehensive results."""
        try:
            initial_state = {
                "question": question,
                "error_count": 0,
                "plot_needed": False,
                "plot_paths": [],
                "schema_info": "",
                "sql_query": "",
                "query_result": "",
                "formatted_answer": "",
                "messages": [],
            }

            # Run the agent
            result = self.graph.invoke(initial_state)

            return {
                "answer": result.get("formatted_answer", "Không thể tạo câu trả lời"),
                "sql_query": result.get("sql_query", ""),
                "query_result": result.get("query_result", ""),
                "plot_paths": result.get("plot_paths", []),
                "source": "selling_database",
            }

        except Exception as e:
            logger.error(f"Error in SQL agent: {e}")
            return {
                "answer": f"Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi: {str(e)}",
                "source": "sql_database",
                "error": str(e),
            }

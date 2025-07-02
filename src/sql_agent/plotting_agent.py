# src/sql_agent/plotting_agent.py

import json
import ast
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
from ..core.llm import get_llm
import logging

logger = logging.getLogger(__name__)


class PlottingAgent:
    """Agent for creating visualizations from SQL query results."""

    def __init__(self, output_dir: str = "plots"):
        self.llm = get_llm()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_plots(
        self, question: str, sql_query: str, query_result: str
    ) -> List[str]:
        """Create appropriate plots based on the data."""
        try:
            # Parse the query result into a DataFrame
            df = self._parse_query_result(query_result)

            if df is None or df.empty:
                logger.warning("No data to plot")
                return []

            # Generate plot code using LLM
            plot_code = self._generate_plot_code(question, sql_query, df)

            if not plot_code:
                logger.warning("No plot code generated")
                return []

            # Execute the plot code and save
            plot_paths = self._execute_plot_code(plot_code, df, question)

            return plot_paths

        except Exception as e:
            logger.error(f"Error creating plots: {e}")
            return []

    def _parse_query_result(self, query_result: str) -> Optional[pd.DataFrame]:
        """Parse SQL query result string into DataFrame."""
        try:
            # Clean the query result
            query_result = query_result.strip()

            # Handle different formats of SQL results
            if query_result.startswith("[") and query_result.endswith("]"):
                # Try to parse as list of tuples/dictionaries
                try:
                    data = ast.literal_eval(query_result)
                    if isinstance(data, list) and len(data) > 0:
                        if isinstance(data[0], dict):
                            return pd.DataFrame(data)
                        elif isinstance(data[0], (list, tuple)):
                            # Handle list of tuples - need to infer column names
                            return pd.DataFrame(
                                data, columns=[f"col_{i}" for i in range(len(data[0]))]
                            )
                except (ValueError, SyntaxError) as e:
                    logger.warning(f"Failed to parse as literal: {e}")

            # Try to parse line by line for formatted output
            lines = query_result.split("\n")
            data_rows = []

            for line in lines:
                line = line.strip()
                if line and not line.startswith("(") and "(" in line:
                    # Extract tuple-like data
                    match = re.search(r"\((.*?)\)", line)
                    if match:
                        values_str = match.group(1)
                        # Parse individual values
                        values = []
                        for val in values_str.split(","):
                            val = val.strip().strip("'\"")
                            # Try to convert to appropriate type
                            try:
                                if "." in val:
                                    values.append(float(val))
                                else:
                                    values.append(int(val))
                            except ValueError:
                                values.append(val)
                        data_rows.append(values)
                elif line.startswith("(") and line.endswith(")"):
                    # Direct tuple format
                    try:
                        values = ast.literal_eval(line)
                        data_rows.append(list(values))
                    except:
                        continue

            if data_rows:
                # Create DataFrame with generic column names
                max_cols = max(len(row) for row in data_rows) if data_rows else 0
                columns = [f"column_{i+1}" for i in range(max_cols)]

                # Pad shorter rows
                for row in data_rows:
                    while len(row) < max_cols:
                        row.append(None)

                df = pd.DataFrame(data_rows, columns=columns)

                # Try to infer better column names from the SQL query
                df = self._infer_column_names(df, sql_query)
                return df

            return None

        except Exception as e:
            logger.error(f"Error parsing query result: {e}")
            return None

    def _infer_column_names(self, df: pd.DataFrame, sql_query: str) -> pd.DataFrame:
        """Try to infer column names from SQL query."""
        try:
            # Extract SELECT clause
            select_match = re.search(
                r"SELECT\s+(.*?)\s+FROM", sql_query, re.IGNORECASE | re.DOTALL
            )
            if select_match:
                select_clause = select_match.group(1)

                # Parse column names/aliases
                columns = []
                for col in select_clause.split(","):
                    col = col.strip()
                    # Check for alias (AS keyword or space)
                    if " AS " in col.upper():
                        alias = col.split(" AS ")[-1].strip()
                        columns.append(alias)
                    elif " " in col and not any(
                        func in col.upper()
                        for func in ["SUM", "COUNT", "AVG", "MAX", "MIN"]
                    ):
                        # Likely an alias without AS
                        parts = col.split()
                        if len(parts) > 1:
                            columns.append(parts[-1])
                        else:
                            columns.append(col)
                    else:
                        # Clean column name
                        col_clean = re.sub(r".*\.", "", col)  # Remove table prefix
                        columns.append(col_clean)

                # Apply new column names if we have the right number
                if len(columns) == len(df.columns):
                    df.columns = columns

        except Exception as e:
            logger.warning(f"Could not infer column names: {e}")

        return df

    def _generate_plot_code(
        self, question: str, sql_query: str, df: pd.DataFrame
    ) -> str:
        """Generate Plotly code for visualization."""
        try:
            # Create metadata about the DataFrame
            df_info = {
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "shape": df.shape,
                "sample_data": df.head(3).to_dict("records") if len(df) > 0 else [],
                "numeric_columns": df.select_dtypes(
                    include=["number"]
                ).columns.tolist(),
                "text_columns": df.select_dtypes(
                    include=["object", "string"]
                ).columns.tolist(),
            }

            prompt = f"""
            Tạo mã Python sử dụng Plotly để vẽ biểu đồ cho dữ liệu sau:
            
            Câu hỏi: {question}
            SQL Query: {sql_query}
            
            Thông tin DataFrame:
            - Columns: {df_info['columns']}
            - Data types: {df_info['dtypes']}
            - Shape: {df_info['shape']}
            - Numeric columns: {df_info['numeric_columns']}
            - Text columns: {df_info['text_columns']}
            - Sample data: {df_info['sample_data']}
            
            Hướng dẫn:
            - Sử dụng plotly.express hoặc plotly.graph_objects
            - DataFrame có sẵn với tên 'df'
            - Tạo 1-2 biểu đồ phù hợp nhất cho dữ liệu
            - Đặt tiêu đề và nhãn bằng tiếng Việt
            - Nếu có dữ liệu doanh thu theo thời gian, tạo biểu đồ đường hoặc cột
            - Nếu có dữ liệu so sánh, tạo biểu đồ cột
            - Trả về danh sách các figure objects
            - Không sử dụng fig.show()
            - Đảm bảo xử lý trường hợp dữ liệu rỗng
            
            Trả về code Python:
            ```python
            import plotly.express as px
            import plotly.graph_objects as go
            
            figures = []
            
            # Kiểm tra dữ liệu không rỗng
            if not df.empty:
                # Tạo biểu đồ phù hợp
                fig = ...
                fig.update_layout(title="...", xaxis_title="...", yaxis_title="...")
                figures.append(fig)
            
            # Return the figures
            figures
            ```
            """

            response = self.llm.invoke(prompt)
            return self._extract_python_code(response.content)

        except Exception as e:
            logger.error(f"Error generating plot code: {e}")
            return ""

    def _extract_python_code(self, response: str) -> str:
        """Extract Python code from LLM response."""
        # Look for code blocks
        pattern = r"```python\n(.*?)```"
        match = re.search(pattern, response, re.DOTALL)

        if match:
            return match.group(1).strip()

        # Fallback: look for any code block
        pattern = r"```\n(.*?)```"
        match = re.search(pattern, response, re.DOTALL)

        if match:
            return match.group(1).strip()

        return response.strip()

    def _execute_plot_code(
        self, plot_code: str, df: pd.DataFrame, question: str
    ) -> List[str]:
        """Execute the plot code and save figures."""
        try:
            # Prepare execution environment
            exec_globals = {
                "pd": pd,
                "px": px,
                "go": go,
                "df": df,
                "figures": [],
                "print": lambda *args: None,  # Suppress print statements
            }

            # Execute the code
            exec(plot_code, exec_globals)

            # Get the figures
            figures = exec_globals.get("figures", [])

            if not figures:
                logger.warning("No figures generated from plot code")
                return []

            # Save figures
            plot_paths = []
            base_filename = re.sub(r"[^\w\s-]", "", question.replace(" ", "_"))[:50]

            for i, fig in enumerate(figures):
                if hasattr(fig, "write_html") and hasattr(fig, "update_layout"):
                    try:
                        # Ensure the figure has proper layout
                        fig.update_layout(
                            font=dict(size=12), margin=dict(l=50, r=50, t=50, b=50)
                        )

                        # Save as HTML for interactivity
                        html_path = self.output_dir / f"{base_filename}_plot_{i+1}.html"
                        fig.write_html(str(html_path))
                        plot_paths.append(str(html_path))

                        # Try to save as PNG if possible
                        try:
                            png_path = (
                                self.output_dir / f"{base_filename}_plot_{i+1}.png"
                            )
                            fig.write_image(str(png_path), width=800, height=600)
                            plot_paths.append(str(png_path))
                        except Exception as e:
                            logger.warning(
                                f"Could not save PNG (install kaleido for PNG export): {e}"
                            )

                    except Exception as e:
                        logger.error(f"Error saving figure {i+1}: {e}")

            return plot_paths

        except Exception as e:
            logger.error(f"Error executing plot code: {e}")
            # Log the problematic code for debugging
            logger.debug(f"Problematic plot code:\n{plot_code}")
            return []

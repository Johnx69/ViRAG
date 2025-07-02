# src/sql_agent/sql_tools.py

import sqlite3
from typing import List, Dict, Any, Optional, Type
from pathlib import Path
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr
import logging

logger = logging.getLogger(__name__)


class SQLQueryInput(BaseModel):
    """Input for SQL query execution."""
    query: str = Field(description="SQL query to execute")


class SQLSchemaInput(BaseModel):
    """Input for getting table schema."""
    table_names: str = Field(description="Comma-separated list of table names")


class SQLListTablesInput(BaseModel):
    """Input for listing tables (no parameters needed)."""
    pass


class SQLQueryTool(BaseTool):
    """Tool for executing SQL queries."""

    name: str = "sql_db_query"
    description: str = """Execute a SQL query against the database.
    Input should be a valid SQL query.
    Returns the query results or an error message if the query fails."""
    args_schema: Type[BaseModel] = SQLQueryInput
    
    _db_path: Path = PrivateAttr()

    def __init__(self, db_path: str, **kwargs):
        super().__init__(**kwargs)
        self._db_path = Path(db_path)
        if not self._db_path.exists():
            raise FileNotFoundError(f"Database file not found: {db_path}")

    def _run(self, query: str) -> str:
        try:
            # Clean the query
            query = query.strip()
            if query.endswith(';'):
                query = query[:-1]

            with sqlite3.connect(self._db_path) as conn:
                # Set row factory to return dictionaries for better parsing
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query)

                # Handle different types of queries
                if query.strip().upper().startswith(("SELECT", "WITH", "PRAGMA")):
                    results = cursor.fetchall()
                    if not results:
                        return "Query executed successfully but returned no results."

                    # Convert to list of dictionaries for easier parsing
                    formatted_results = []
                    for row in results:
                        formatted_results.append(dict(row))

                    return str(formatted_results)
                else:
                    # For non-SELECT queries
                    conn.commit()
                    return f"Query executed successfully. Rows affected: {cursor.rowcount}"

        except sqlite3.Error as e:
            logger.error(f"SQL execution error: {e}")
            return f"Error executing query: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return f"Unexpected error: {str(e)}"


class SQLSchemaTool(BaseTool):
    """Tool for getting table schemas."""

    name: str = "sql_db_schema"
    description: str = """Get the schema and sample rows for specified tables.
    Input should be a comma-separated list of table names."""
    args_schema: Type[BaseModel] = SQLSchemaInput
    
    _db_path: Path = PrivateAttr()

    def __init__(self, db_path: str, **kwargs):
        super().__init__(**kwargs)
        self._db_path = Path(db_path)

    def _run(self, table_names: str) -> str:
        try:
            tables = [name.strip() for name in table_names.split(",")]
            result = []

            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()

                for table in tables:
                    # Get table schema
                    cursor.execute(f"PRAGMA table_info({table})")
                    columns = cursor.fetchall()

                    if not columns:
                        result.append(f"Table '{table}' not found.")
                        continue

                    # Format schema
                    schema_info = f"\nTable: {table}\n"
                    schema_info += "Columns:\n"
                    for col in columns:
                        schema_info += f"  - {col[1]} ({col[2]})"
                        if col[3]:  # NOT NULL
                            schema_info += " NOT NULL"
                        if col[5]:  # PRIMARY KEY
                            schema_info += " PRIMARY KEY"
                        schema_info += "\n"

                    # Get sample rows
                    cursor.execute(f"SELECT * FROM {table} LIMIT 3")
                    sample_rows = cursor.fetchall()

                    if sample_rows:
                        column_names = [desc[0] for desc in cursor.description]
                        schema_info += f"\nSample rows from {table}:\n"
                        for row in sample_rows:
                            row_dict = dict(zip(column_names, row))
                            schema_info += f"  {row_dict}\n"

                    result.append(schema_info)

            return "\n".join(result)

        except Exception as e:
            logger.error(f"Error getting schema: {e}")
            return f"Error getting schema: {str(e)}"


class SQLListTablesTool(BaseTool):
    """Tool for listing all tables in the database."""

    name: str = "sql_db_list_tables"
    description: str = "List all tables in the database."
    args_schema: Type[BaseModel] = SQLListTablesInput
    
    _db_path: Path = PrivateAttr()

    def __init__(self, db_path: str, **kwargs):
        super().__init__(**kwargs)
        self._db_path = Path(db_path)

    def _run(self) -> str:
        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()

                if not tables:
                    return "No tables found in the database."

                table_names = [table[0] for table in tables]
                return f"Available tables: {', '.join(table_names)}"

        except Exception as e:
            logger.error(f"Error listing tables: {e}")
            return f"Error listing tables: {str(e)}"


class SQLToolkit:
    """Collection of SQL tools for the agent."""

    def __init__(self, db_path: str, llm):
        self.db_path = db_path
        self.llm = llm
        self._tools = None

    def get_tools(self) -> List[BaseTool]:
        """Get all SQL tools."""
        if self._tools is None:
            self._tools = [
                SQLQueryTool(self.db_path),
                SQLSchemaTool(self.db_path),
                SQLListTablesTool(self.db_path),
            ]
        return self._tools
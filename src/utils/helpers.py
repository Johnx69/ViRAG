import re
from typing import List, Dict, Any


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove special characters but keep Vietnamese characters
    text = re.sub(r"[^\w\s\u00C0-\u024F\u1E00-\u1EFF]", " ", text)

    return text.strip()


def truncate_text(text: str, max_length: int = 1000) -> str:
    """Truncate text to maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def format_response(response: Dict[str, Any]) -> str:
    """Format response for display."""
    answer = response.get("answer", "")
    source = response.get("source", "unknown")
    confidence = response.get("confidence_score")

    formatted = f"Trả lời: {answer}\n\n"
    formatted += f"Nguồn: {source}"

    if confidence is not None:
        formatted += f" (Độ tin cậy: {confidence:.2f})"

    num_sources = response.get("num_sources")
    if num_sources:
        formatted += f"\nSố nguồn tham khảo: {num_sources}"

    return formatted


def validate_query(query: str) -> bool:
    """Validate if query is appropriate for processing."""
    if not query or not query.strip():
        return False

    if len(query.strip()) < 3:
        return False

    return True

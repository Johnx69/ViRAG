# gradio_app.py

import gradio as gr
import sys
from pathlib import Path
import json
import time
import base64
import os
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.pipeline.enhanced_rag_pipeline import EnhancedVietnameseRAGPipeline
from src.utils.logging import setup_logging
from src.config.settings import settings

# Setup logging
setup_logging(log_level="INFO")

# Global pipeline instance
pipeline = None


def initialize_pipeline():
    """Initialize the RAG pipeline."""
    global pipeline
    if pipeline is None:
        pipeline = EnhancedVietnameseRAGPipeline()
    return pipeline


def format_confidence_score(score: float) -> str:
    """Format confidence score with emoji and description."""
    if score >= 0.7:
        return f"🟢 {score:.3f} (Cao)"
    elif score >= 0.4:
        return f"🟡 {score:.3f} (Trung bình)"
    else:
        return f"🔴 {score:.3f} (Thấp)"


def format_source(source: str) -> str:
    """Format source with emoji."""
    source_emoji = {
        "wikipedia": "📚 Wikipedia",
        "web_search": "🌐 Web Search",
        "selling_database": "💾 SQL Database",
        "error": "❌ Lỗi",
    }
    return source_emoji.get(source, f"❓ {source}")


def format_retrieved_docs(docs: List[Dict[str, Any]]) -> str:
    """Format retrieved documents as HTML."""
    if not docs:
        return "<p>Không có tài liệu nào được truy xuất.</p>"

    html = "<div style='max-height: 400px; overflow-y: auto;'>"
    for i, doc in enumerate(docs, 1):
        score_color = (
            "#28a745"
            if doc.get("score", 0) >= 0.7
            else "#ffc107" if doc.get("score", 0) >= 0.4 else "#dc3545"
        )

        html += f"""
        <div style="border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 8px; background: #f8f9fa;">
            <h4 style="color: #2c3e50; margin: 0 0 10px 0;">📄 {doc.get('title', f'Tài liệu {i}')}</h4>
            <p style="margin: 5px 0;"><strong>📝 Nội dung:</strong> {doc.get('content', 'Không có nội dung')}</p>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px;">
                <span style="color: {score_color}; font-weight: bold;">⭐ Điểm: {doc.get('score', 0.0):.3f}</span>
                {f'<a href="{doc.get("url", "")}" target="_blank" style="color: #007bff; text-decoration: none;">🔗 Xem nguồn</a>' if doc.get('url') else ''}
            </div>
        </div>
        """
    html += "</div>"
    return html


def format_process_log(process_log: List[str]) -> str:
    """Format process log as HTML."""
    if not process_log:
        return "<p>Không có nhật ký xử lý.</p>"

    html = "<div style='background: #f0f2f6; padding: 15px; border-radius: 8px; border-left: 4px solid #007bff;'>"
    html += "<h4 style='color: #007bff; margin: 0 0 10px 0;'>📋 Nhật ký xử lý:</h4>"

    for i, step in enumerate(process_log, 1):
        html += f"<div style='margin: 5px 0;'><strong>{i}.</strong> {step}</div>"

    html += "</div>"
    return html


def format_sql_info(result: Dict[str, Any]) -> str:
    """Format SQL information as HTML."""
    html = ""

    if result.get("sql_query"):
        html += f"""
        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #28a745; margin: 10px 0;">
            <h4 style="color: #28a745; margin: 0 0 10px 0;">🔍 SQL Query:</h4>
            <code style="background: #e9ecef; padding: 10px; border-radius: 4px; display: block; font-family: 'Courier New', monospace;">
                {result['sql_query']}
            </code>
        </div>
        """

    if result.get("query_result"):
        html += f"""
        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #17a2b8; margin: 10px 0;">
            <h4 style="color: #17a2b8; margin: 0 0 10px 0;">📊 Kết quả truy vấn:</h4>
            <details>
                <summary style="cursor: pointer; color: #007bff;">Xem chi tiết</summary>
                <pre style="background: #e9ecef; padding: 10px; border-radius: 4px; margin-top: 10px; overflow-x: auto;">
{result['query_result']}
                </pre>
            </details>
        </div>
        """

    return html


def image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
            return encoded_string
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return ""


def create_plot_html(plot_paths: List[str]) -> str:
    """Create HTML for displaying plots with fixed image handling."""
    if not plot_paths:
        return "<p>Không có biểu đồ nào được tạo.</p>"

    html = "<h4>📈 Biểu đồ được tạo:</h4>"

    for i, plot_path in enumerate(plot_paths, 1):
        plot_path = Path(plot_path)

        if plot_path.exists():
            if plot_path.suffix.lower() == ".html":
                try:
                    with open(plot_path, "r", encoding="utf-8") as f:
                        plot_html = f.read()
                    html += f"""
                    <div style='margin: 20px 0; padding: 20px; border: 2px solid #007bff; border-radius: 10px; background: #f8f9fa;'>
                        <h5 style='color: #007bff; margin-bottom: 15px;'>📊 Biểu đồ {i} (Interactive)</h5>
                        <div style='background: white; padding: 10px; border-radius: 5px;'>
                            {plot_html}
                        </div>
                    </div>
                    """
                except Exception as e:
                    html += f"<p style='color: red;'>❌ Lỗi hiển thị biểu đồ HTML {i}: {e}</p>"

            elif plot_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                try:
                    # Convert image to base64 for embedding
                    base64_image = image_to_base64(str(plot_path))
                    if base64_image:
                        # Determine MIME type
                        mime_type = f"image/{plot_path.suffix[1:].lower()}"
                        if mime_type == "image/jpg":
                            mime_type = "image/jpeg"

                        html += f"""
                        <div style='margin: 20px 0; padding: 20px; border: 2px solid #28a745; border-radius: 10px; background: #f8f9fa;'>
                            <h5 style='color: #28a745; margin-bottom: 15px;'>🖼️ Biểu đồ {i} (Static Image)</h5>
                            <div style='text-align: center; background: white; padding: 10px; border-radius: 5px;'>
                                <img src="data:{mime_type};base64,{base64_image}" 
                                     style='max-width: 100%; height: auto; border-radius: 5px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);' 
                                     alt='Biểu đồ {i}'/>
                            </div>
                            <p style='text-align: center; margin-top: 10px; color: #666; font-size: 0.9em;'>
                                📁 File: {plot_path.name}
                            </p>
                        </div>
                        """
                    else:
                        html += f"<p style='color: orange;'>⚠️ Không thể đọc file ảnh: {plot_path.name}</p>"
                except Exception as e:
                    html += f"<p style='color: red;'>❌ Lỗi hiển thị ảnh {i}: {e}</p>"
        else:
            html += f"<p style='color: orange;'>⚠️ Không thể tìm thấy file biểu đồ: {plot_path}</p>"

    return html


def extract_plot_images(plot_paths: List[str]) -> List[str]:
    """Extract image files from plot paths for Gradio Gallery display."""
    image_paths = []
    for plot_path in plot_paths:
        plot_path = Path(plot_path)
        if plot_path.exists() and plot_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            image_paths.append(str(plot_path))
    return image_paths


def process_query(
    question: str,
    simple_rewrite: bool,
    decompose_rewrite: bool,
    hyde_rewrite: bool,
    max_web_searches: int,
    max_return_docs: int,
    progress=gr.Progress(),
) -> Tuple[str, str, str, str, str, str, str, List[str]]:
    """Process the user query and return formatted results."""

    if not question.strip():
        return "⚠️ Vui lòng nhập câu hỏi!", "", "", "", "", "", "", []

    # Initialize pipeline
    progress(0.1, desc="Khởi tạo hệ thống...")
    pipeline = initialize_pipeline()

    # Prepare rewrite strategies
    strategies = []
    if simple_rewrite:
        strategies.append("simple")
    if decompose_rewrite:
        strategies.append("decompose")
    if hyde_rewrite:
        strategies.append("hyde")

    progress(0.3, desc="Xử lý câu hỏi...")

    # Process query
    start_time = time.time()
    result = pipeline.query(
        question=question,
        rewrite_strategies=strategies if strategies else None,
        max_web_searches=max_web_searches,
        max_return_docs=max_return_docs,
    )
    processing_time = time.time() - start_time

    progress(0.9, desc="Định dạng kết quả...")

    # Format answer
    answer = result.get("answer", "Không có câu trả lời")

    # Format metadata
    metadata = f"""
    **⏱️ Thời gian xử lý:** {processing_time:.2f} giây
    
    **📊 Nguồn dữ liệu:** {format_source(result.get('source', 'unknown'))}
    
    **📈 Tin cậy:** {format_confidence_score(result['confidence_score']) if 'confidence_score' in result else 'N/A'}
    
    **📚 Số nguồn:** {result.get('num_sources', 0)}
    
    **📈 Biểu đồ:** {len(result.get('plot_paths', []))}
    """

    # Format rewrite info
    rewrite_info = ""
    if "rewrite_info" in result:
        info = result["rewrite_info"]
        rewrite_info = f"""
        **🔄 Chiến lược viết lại:** {info['strategy']} (Lần thử {info['attempt']})
        
        **❓ Câu hỏi gốc:** {info['original_query']}
        
        **✏️ Câu hỏi đã viết lại:** {info['rewritten_query']}
        """

    # Format retrieved documents
    retrieved_docs_html = format_retrieved_docs(result.get("retrieved_docs", []))

    # Format process log
    process_log_html = format_process_log(result.get("process_log", []))

    # Format SQL info
    sql_info_html = (
        format_sql_info(result) if result.get("source") == "selling_database" else ""
    )

    # Format plots
    plots_html = create_plot_html(result.get("plot_paths", []))

    # Extract image paths for gallery
    image_paths = extract_plot_images(result.get("plot_paths", []))

    progress(1.0, desc="Hoàn thành!")

    return (
        answer,
        metadata,
        rewrite_info,
        retrieved_docs_html,
        process_log_html,
        sql_info_html,
        plots_html,
        image_paths,
    )


def create_interface():
    """Create the Gradio interface."""

    # Custom CSS
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .answer-box {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .metadata-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
    }
    
    .rewrite-box {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
    }
    """

    with gr.Blocks(
        theme=gr.themes.Soft(), css=css, title="Vietnamese RAG System"
    ) as demo:

        # Header
        gr.Markdown(
            """
        # 🤖 Hệ thống RAG Tiếng Việt Nâng cao
        
        Hệ thống trả lời câu hỏi thông minh với khả năng viết lại câu hỏi và tìm kiếm đa nguồn.
        """
        )

        with gr.Row():
            with gr.Column(scale=2):
                # Input section
                gr.Markdown("## 💬 Đặt câu hỏi")

                question_input = gr.Textbox(
                    label="Câu hỏi của bạn",
                    placeholder="Ví dụ: Việt Nam có bao nhiêu tỉnh thành? hoặc Nhân viên nào bán được nhiều nhất tháng 8?",
                    lines=3,
                )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🔄 Chiến lược viết lại câu hỏi")
                        simple_rewrite = gr.Checkbox(
                            label="Simple - Viết lại đơn giản",
                            info="Tối ưu hóa câu hỏi cho tìm kiếm",
                        )
                        decompose_rewrite = gr.Checkbox(
                            label="Decompose - Phân tách câu hỏi",
                            info="Chia câu hỏi phức tạp thành câu con",
                        )
                        hyde_rewrite = gr.Checkbox(
                            label="HyDE - Tài liệu giả định",
                            info="Tạo văn bản mẫu để cải thiện tìm kiếm",
                        )

                    with gr.Column():
                        gr.Markdown("### ⚙️ Cài đặt nâng cao")
                        max_web_searches = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=3,
                            step=1,
                            label="Số lượng tìm kiếm web tối đa",
                        )
                        max_return_docs = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=5,
                            step=1,
                            label="Số tài liệu trả về tối đa",
                        )

                submit_btn = gr.Button("🚀 Gửi câu hỏi", variant="primary", size="lg")

            with gr.Column(scale=1):
                # System status
                gr.Markdown("## 🔧 Trạng thái hệ thống")

                def check_system_status():
                    try:
                        pipeline = initialize_pipeline()

                        # Check Wikipedia index
                        wiki_status = (
                            "✅ Sẵn sàng" if pipeline.wikipedia_index else "⚠️ Chưa có"
                        )

                        # Check SQL database
                        db_path = settings.ROOT / "data" / "db" / "sell_data.sqlite"
                        sql_status = (
                            "✅ Sẵn sàng" if db_path.exists() else "❌ Không tìm thấy"
                        )

                        return f"""
                        **📚 Wikipedia Index:** {wiki_status}
                        
                        **💾 SQL Database:** {sql_status}
                        
                        **🤖 Generator:** Gemini 2.0 Flash
                        
                        **✏️ Rewriter:** GPT-4o
                        
                        **🔍 Embeddings:** AITeamVN/Vietnamese_Embedding_v2
                        
                        **📑 Reranker:** BAAI/bge-reranker-v2-m3
                        """
                    except Exception as e:
                        return f"❌ Lỗi kiểm tra hệ thống: {str(e)}"

                system_status = gr.Markdown(check_system_status())

                # Refresh status button
                refresh_btn = gr.Button("🔄 Làm mới trạng thái")
                refresh_btn.click(fn=check_system_status, outputs=system_status)

                # Help section
                gr.Markdown(
                    """
                ## 💡 Hướng dẫn
                
                **🔹 Câu hỏi bán hàng:**
                - So sánh doanh thu của mỗi chi nhánh trong 12 tháng 
                - Những sản phẩm nào có giá vốn lớn hơn 100.000.000 đồng?
                - Những khách hàng nào đã mua sản phẩm 'Phần mềm Cube IQ'?
                
                **🔹 Kiến thức chung:**
                - Elaphidion excelsum thuộc họ nào?
                - Quercus trabutii thuộc họ thực vật nào?
                - Ptychocoleus wichurae thuộc họ nào?
                
                **💡 Tip:** Sử dụng kết hợp các chiến lược viết lại để có kết quả tốt nhất!
                """
                )

        # Results section
        gr.Markdown("## 📋 Kết quả")

        with gr.Row():
            with gr.Column():
                # Main answer
                answer_output = gr.Markdown(
                    label="💡 Câu trả lời", elem_classes=["answer-box"]
                )

                # Metadata
                metadata_output = gr.Markdown(
                    label="📊 Thông tin", elem_classes=["metadata-box"]
                )

                # Rewrite information
                rewrite_output = gr.Markdown(
                    label="🔄 Thông tin viết lại",
                    elem_classes=["rewrite-box"],
                    visible=False,
                )

        # Detailed results in tabs
        with gr.Tabs():
            with gr.Tab("📚 Tài liệu truy xuất"):
                retrieved_docs_output = gr.HTML()

            with gr.Tab("📋 Nhật ký xử lý"):
                process_log_output = gr.HTML()

            with gr.Tab("💾 SQL & Database"):
                sql_info_output = gr.HTML()

            with gr.Tab("📈 Biểu đồ & Hình ảnh"):
                with gr.Row():
                    with gr.Column():
                        plots_output = gr.HTML(label="Biểu đồ HTML")
                    with gr.Column():
                        # Add image gallery for better image display
                        image_gallery = gr.Gallery(
                            label="🖼️ Thư viện ảnh",
                            show_label=True,
                            elem_id="image_gallery",
                            columns=2,
                            rows=2,
                            height="auto",
                            allow_preview=True,
                        )

        # Connect the submit function
        def handle_submit(*inputs):
            results = process_query(*inputs)

            # Show/hide rewrite info based on whether it exists
            rewrite_visible = bool(results[2].strip())

            return [
                results[0],  # answer
                results[1],  # metadata
                results[2],  # rewrite_info
                gr.update(visible=rewrite_visible),  # rewrite_output visibility
                results[3],  # retrieved_docs
                results[4],  # process_log
                results[5],  # sql_info
                results[6],  # plots
                results[7],  # image_gallery
            ]

        submit_btn.click(
            fn=handle_submit,
            inputs=[
                question_input,
                simple_rewrite,
                decompose_rewrite,
                hyde_rewrite,
                max_web_searches,
                max_return_docs,
            ],
            outputs=[
                answer_output,
                metadata_output,
                rewrite_output,
                rewrite_output,  # For visibility update
                retrieved_docs_output,
                process_log_output,
                sql_info_output,
                plots_output,
                image_gallery,  # New output for image gallery
            ],
        )

        # Also trigger on Enter key
        question_input.submit(
            fn=handle_submit,
            inputs=[
                question_input,
                simple_rewrite,
                decompose_rewrite,
                hyde_rewrite,
                max_web_searches,
                max_return_docs,
            ],
            outputs=[
                answer_output,
                metadata_output,
                rewrite_output,
                rewrite_output,
                retrieved_docs_output,
                process_log_output,
                sql_info_output,
                plots_output,
                image_gallery,
            ],
        )

    return demo


if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()

    # Launch with custom settings
    demo.launch(
        share=False,  # Set to True if you want to create a public link
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,
        show_error=True,
        max_threads=4,
    )

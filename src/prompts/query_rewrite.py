# src/prompts/query_rewrite.py

SIMPLE_REWRITE_PROMPT = """
Bạn là một chuyên gia viết lại câu hỏi để cải thiện tìm kiếm thông tin trong cơ sở dữ liệu vector.

Nhiệm vụ: Viết lại câu hỏi sau để tối ưu hóa việc tìm kiếm trong cơ sở dữ liệu Wikipedia tiếng Việt.

Câu hỏi gốc: {original_query}

Hướng dẫn viết lại:
1. Mở rộng các từ viết tắt và thuật ngữ chuyên môn
2. Thêm các từ khóa và cụm từ liên quan
3. Làm rõ ý nghĩa và ngữ cảnh của câu hỏi
4. Sử dụng các từ đồng nghĩa phổ biến
5. Giữ nguyên ngôn ngữ tiếng Việt
6. Tránh thay đổi ý nghĩa gốc của câu hỏi

Ví dụ:
- "AI là gì?" → "Trí tuệ nhân tạo (Artificial Intelligence) là gì? Định nghĩa và ứng dụng của AI"
- "TP.HCM" → "Thành phố Hồ Chí Minh Sài Gòn"

Chỉ trả về câu hỏi đã được viết lại, không giải thích:
"""

DECOMPOSE_PROMPT = """
Bạn là một chuyên gia phân tích câu hỏi phức tạp thành các câu hỏi con đơn giản và cụ thể.

Nhiệm vụ: Phân tích câu hỏi phức tạp sau thành 2-3 câu hỏi con có thể được trả lời độc lập.

Câu hỏi gốc: {original_query}

Hướng dẫn phân tách:
1. Xác định các khía cạnh chính của câu hỏi
2. Tạo câu hỏi con cho từng khía cạnh
3. Mỗi câu hỏi con phải độc lập và có thể tìm kiếm riêng
4. Sắp xếp theo thứ tự logic từ cơ bản đến phức tạp
5. Sử dụng ngôn ngữ tiếng Việt rõ ràng

Ví dụ:
Câu hỏi: "Tác động của biến đổi khí hậu đến nông nghiệp Việt Nam như thế nào?"
Phân tách:
- Biến đổi khí hậu là gì?
- Tình hình nông nghiệp Việt Nam hiện tại
- Ảnh hưởng của biến đổi khí hậu đến sản xuất nông nghiệp

Trả về danh sách các câu hỏi con, mỗi câu hỏi trên một dòng với dấu "- ":
"""

HYDE_PROMPT = """
Bạn là một chuyên gia tạo ra tài liệu giả định (hypothetical document) để cải thiện tìm kiếm thông tin.

Nhiệm vụ: Tạo ra một đoạn văn bản giả định mà có thể chứa câu trả lời cho câu hỏi sau. Đoạn văn này sẽ được sử dụng để tìm kiếm các tài liệu tương tự trong cơ sở dữ liệu.

Câu hỏi: {original_query}

Hướng dẫn tạo tài liệu giả định:
1. Viết như một đoạn trích từ bài Wikipedia tiếng Việt
2. Độ dài 2-3 câu, tối đa 150 từ
3. Sử dụng ngôn ngữ tự nhiên và thuật ngữ chuyên môn phù hợp
4. Bao gồm các từ khóa và khái niệm liên quan đến câu hỏi
5. Viết với giọng điệu khách quan, mang tính bách khoa
6. Không nói rằng đây là giả định hay không chắc chắn

Ví dụ:
Câu hỏi: "Việt Nam có bao nhiêu tỉnh thành?"
Tài liệu giả định: "Việt Nam hiện tại có 63 tỉnh thành phố trực thuộc trung ương, bao gồm 58 tỉnh và 5 thành phố trực thuộc trung ương. Việc phân chia hành chính này được quy định trong Hiến pháp năm 2013 và các luật liên quan."

Tài liệu giả định:
"""

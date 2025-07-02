import os
from dotenv import load_dotenv
from src import VietnameseRAGPipeline, setup_logging, format_response, validate_query
import time


def main():
    # Load environment variables
    load_dotenv()

    # Setup logging
    setup_logging(log_level="INFO", log_file="logs/rag_system.log")

    # Initialize the RAG pipeline
    print("Initializing Vietnamese RAG System with SQL Agent...")
    rag_pipeline = VietnameseRAGPipeline()

    # Option to index Wikipedia data (run once)
    index_data = input("Do you want to index Wikipedia data? (y/n): ").lower() == "y"
    if index_data:
        num_samples = input("Enter number of samples to index (press Enter for all): ")
        num_samples = int(num_samples) if num_samples.strip() else None

        print("Indexing Wikipedia data...")
        start_time = time.time()  # Start timer

        rag_pipeline.index_wikipedia_data(num_samples)

        end_time = time.time()  # End timer
        print("Indexing completed!")
        print(f"Time taken: {end_time - start_time:.2f} seconds")

    # Interactive query loop
    print("\nVietnamese RAG System with SQL Agent Ready!")
    print("Examples:")
    print("- Doanh thu tháng 8 là bao nhiêu?")
    print("- Nhân viên nào bán được nhiều nhất?")
    print("- Chi nhánh nào có doanh thu cao nhất?")
    print("- What is artificial intelligence? (general knowledge)")
    print("Type 'quit' to exit")
    print("-" * 50)

    while True:
        query = input("\nNhập câu hỏi của bạn: ").strip()

        if query.lower() == "quit":
            break

        if not validate_query(query):
            print("Vui lòng nhập một câu hỏi hợp lệ.")
            continue

        try:
            print("\nĐang xử lý...")
            response = rag_pipeline.query(query)
            print("\n" + "=" * 50)
            print(format_response(response))

            # Show plot information if available
            if response.get("plot_paths"):
                print(f"\nBiểu đồ đã được lưu tại:")
                for path in response["plot_paths"]:
                    print(f"  - {path}")

            print("=" * 50)

        except Exception as e:
            print(f"Lỗi: {e}")


if __name__ == "__main__":
    main()

import weaviate
import re
from dotenv import load_dotenv
import os    
# Load environment
load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
def test_connection():
    try:
        # Parse URL
        url_pattern = r'http://([^:]+):(\d+)'
        match = re.match(url_pattern, WEAVIATE_URL)
        if not match:
            raise ValueError(f"Invalid URL format: {WEAVIATE_URL}")
        
        host = match.group(1)
        port = int(match.group(2))
        
        # Connect
        client = weaviate.connect_to_custom(
            http_host=host,
            http_port=port,
            http_secure=False,
            grpc_host=host,
            grpc_port=50051,
            grpc_secure=False,
            auth_credentials=weaviate.auth.AuthApiKey(WEAVIATE_API_KEY),
        )
        
        print("✅ Connection successful!")
        print(f"✅ Weaviate is ready: {client.is_ready()}")
        
        # Test basic operations
        collections = client.collections.list_all()
        print(f"✅ Available collections: {len(collections)}")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    test_connection()

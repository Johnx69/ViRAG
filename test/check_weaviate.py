#!/usr/bin/env python3
"""
Quick Weaviate Status Check
Run this script to quickly check your Weaviate database status.
"""

import os
import weaviate
import re
from dotenv import load_dotenv

def quick_status_check():
    """Quick check of Weaviate status and object counts."""
    
    # Load environment
    load_dotenv()
    
    WEAVIATE_URL = os.getenv("WEAVIATE_URL")
    WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
    
    if not WEAVIATE_URL or not WEAVIATE_API_KEY:
        print("❌ Missing WEAVIATE_URL or WEAVIATE_API_KEY in environment")
        return
    
    try:
        # Parse URL
        url_pattern = r'http://([^:]+):(\d+)'
        match = re.match(url_pattern, WEAVIATE_URL)
        if not match:
            print(f"❌ Invalid URL format: {WEAVIATE_URL}")
            return
        
        host = match.group(1)
        port = int(match.group(2))
        
        print(f"🔌 Connecting to Weaviate at {host}:{port}...")
        
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
        
        print("✅ Connected successfully!")
        print(f"✅ Weaviate ready: {client.is_ready()}")
        
        # List collections
        collections = client.collections.list_all()
        
        if not collections:
            print("📝 No collections found")
            client.close()
            return
        
        print(f"\n📊 COLLECTIONS SUMMARY:")
        print("=" * 40)
        
        total_objects = 0
        for collection_name in collections:
            try:
                collection = client.collections.get(collection_name)
                result = collection.aggregate.over_all(total_count=True)
                count = result.total_count if result.total_count else 0
                total_objects += count
                
                print(f"📁 {collection_name}: {count:,} objects")
                
                # Special check for Wikipedia collection
                if collection_name.lower() == "vietnamesewikipedia":
                    print(f"   👉 This is your Wikipedia collection!")
                    if count > 0:
                        print(f"   📈 You can resume indexing from document #{count}")
                    
            except Exception as e:
                print(f"❌ Error checking {collection_name}: {e}")
        
        print("=" * 40)
        print(f"📈 TOTAL OBJECTS: {total_objects:,}")
        
        client.close()
        print("🔌 Connection closed")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    print("🚀 QUICK WEAVIATE STATUS CHECK")
    print("=" * 30)
    quick_status_check()
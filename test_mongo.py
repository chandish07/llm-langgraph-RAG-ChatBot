from pymongo import MongoClient

def test_connection():
    try:
        client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        print("MongoDB connection successful!")
        
        # Test database access
        db = client["TMdata"]
        collection = db["sample1"]
        doc_count = collection.count_documents({})
        print(f"Found {doc_count} documents in TMdata.sample1")
        
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
    finally:
        if 'client' in locals():
            client.close()

if __name__ == "__main__":
    test_connection() 
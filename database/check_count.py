from pymilvus import MilvusClient

try:
    # Kết nối Milvus
    client = MilvusClient(uri="http://localhost:19530", db_name="default")
    
    # Kiểm tra số lượng
    res = client.query(
        collection_name="AIC_2024_1",
        filter="",
        output_fields=["count(*)"]
    )
    
    print("="*30)
    print(f"📊 TỔNG SỐ VECTOR: {res[0]['count(*)']}")
    print("="*30)

except Exception as e:
    print(f"❌ LỖI KẾT NỐI/QUERY: {e}")
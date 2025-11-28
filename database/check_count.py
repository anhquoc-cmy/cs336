from pymilvus import MilvusClient

# Kết nối
client = MilvusClient(uri="http://localhost:19530", db_name="default")

# Kiểm tra số lượng
res = client.query(
    collection_name="AIC_2024_1",
    filter="",
    output_fields=["count(*)"]
)

print(f"📊 Đang có: {res[0]['count(*)']} vector trong Database.")
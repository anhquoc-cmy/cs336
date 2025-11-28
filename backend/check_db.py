from pymilvus import MilvusClient
import numpy as np

# Cấu hình
URI = "http://localhost:19530"
COLLECTION_NAME = "AIC_2024_1"

client = MilvusClient(uri=URI, db_name="default")

def debug():
    print(f"🔍 Đang kiểm tra Collection: {COLLECTION_NAME}")
    
    # 1. Load Collection thủ công (để chắc chắn nó tìm được)
    client.load_collection(COLLECTION_NAME)
    print("✅ Đã Load Collection lên RAM.")

    # 2. Lấy thử 1 dòng dữ liệu xem Vector có bị lỗi không
    res = client.query(
        collection_name=COLLECTION_NAME,
        filter="id >= 0", # Lấy dòng đầu tiên
        output_fields=["embedding", "path"],
        limit=1
    )
    
    if not res:
        print("❌ LỖI: Không lấy được dòng nào (Dù count báo có).")
        return

    vector = res[0]['embedding']
    path = res[0]['path']
    print(f"📸 Ảnh mẫu: {path}")
    print(f"   Vector (5 số đầu): {vector[:5]}")
    
    # Kiểm tra vector có phải toàn số 0 không
    if all(v == 0 for v in vector):
        print("❌ LỖI NGHIÊM TRỌNG: Vector toàn số 0! (Do lỗi Embedding/Ảnh đen)")
        return
    else:
        print("✅ Vector trông có vẻ ổn (Khác 0).")

    # 3. Thử Search chính cái vector đó (Tìm chính nó phải ra)
    print("\n🔎 Đang thử Search chính vector này...")
    search_res = client.search(
        collection_name=COLLECTION_NAME,
        data=[vector],
        limit=5,
        search_params={"metric_type": "IP", "params": {"nprobe": 128}},
        output_fields=["path"]
    )
    
    if search_res and search_res[0]:
        print(f"✅ Search thành công! Tìm thấy {len(search_res[0])} kết quả.")
        print("   Top 1:", search_res[0][0]['entity']['path'])
    else:
        print("❌ LỖI: Search trả về rỗng!")

if __name__ == "__main__":
    try:
        debug()
    except Exception as e:
        print(f"❌ Lỗi Crash: {e}")
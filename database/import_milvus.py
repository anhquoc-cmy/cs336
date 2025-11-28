import os
import glob
import pandas as pd
from pymilvus import MilvusClient, DataType

# ================= CẤU HÌNH =================
MILVUS_URI = "http://localhost:19530" 
DB_NAME = "default"
COLLECTION_NAME = "AIC_2024_1"
DIMENSION = 1024  # ViT-H-14-378

# ⚠️ QUAN TRỌNG: 
# Set = True  : Nếu chạy LẦN ĐẦU (sẽ xóa sạch dữ liệu cũ để tạo mới)
# Set = False : Nếu chạy LẦN 2 trở đi (để nạp tiếp Batch 2, 3... vào mà không mất Batch 1)
RESET_DB = False 

DATA_DIR = "./data_for_milvus" # Folder chứa file parquet

def import_data_to_milvus():
    print(f"Connecting to Milvus at {MILVUS_URI}...")
    client = MilvusClient(uri=MILVUS_URI, db_name=DB_NAME)
    
    # --- BƯỚC 1: XỬ LÝ COLLECTION ---
    if RESET_DB:
        # Nếu chọn Reset, xóa collection cũ đi làm lại
        if client.has_collection(COLLECTION_NAME):
            print(f"⚠️ WARNING: Dropping collection {COLLECTION_NAME} because RESET_DB=True")
            client.drop_collection(COLLECTION_NAME)
        
        # Tạo Schema mới (Chỉ tạo khi Reset)
        print("Creating new schema and collection...")
        schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
        
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=DIMENSION)
        
        # Metadata fields (Khớp Backend)
        schema.add_field(field_name="video", datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(field_name="frame_id", datatype=DataType.INT64)
        schema.add_field(field_name="time", datatype=DataType.FLOAT)
        schema.add_field(field_name="path", datatype=DataType.VARCHAR, max_length=512)
        # Thêm trường ASR
        
        # Index Params
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="embedding", 
            index_type="IVF_FLAT", # IVF_FLAT cân bằng tốt, nếu RAM dư dả (32GB+) có thể dùng HNSW cho nhanh hơn
            metric_type="IP",      # Inner Product (Quan trọng cho CLIP)
            params={"nlist": 1024}
        )

        client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params
        )
        print(f"✅ Collection {COLLECTION_NAME} created.")
        
    else:
        # Nếu không Reset (chạy Batch 2, 3...), chỉ kiểm tra xem có Collection chưa
        if not client.has_collection(COLLECTION_NAME):
            print("❌ Error: Collection not found! Please run with RESET_DB = True for the first batch.")
            return
        print(f"ℹ️ Appending data to existing collection {COLLECTION_NAME}...")

    # --- BƯỚC 2: INSERT DATA ---
    parquet_files = glob.glob(os.path.join(DATA_DIR, "*.parquet"))
    print(f"📂 Found {len(parquet_files)} parquet files to insert.")
    
    if len(parquet_files) == 0:
        print("⚠️ No parquet files found. Check your DATA_DIR.")
        return

    total_inserted = 0
    for file_path in parquet_files:
        try:
            print(f"Inserting {os.path.basename(file_path)}...", end=" ")
            df = pd.read_parquet(file_path)
            data = df.to_dict('records')
            
            res = client.insert(collection_name=COLLECTION_NAME, data=data)
            count = res['insert_count']
            total_inserted += count
            print(f"✅ OK ({count} vectors)")
            
        except Exception as e:
            print(f"\n❌ Failed to insert {file_path}: {e}")

    print("="*30)
    print(f"🎉 DONE! Total vectors inserted: {total_inserted}")
    print("="*30)

if __name__ == "__main__":
    import_data_to_milvus()
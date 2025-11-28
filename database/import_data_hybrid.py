import os
import glob
import pandas as pd
from pymilvus import MilvusClient, DataType

# ================= CẤU HÌNH =================
# 🟢 ĐÃ SỬA: Đổi về cổng mặc định của Milvus
MILVUS_URI = "http://localhost:19530" 
DB_NAME = "default"
COLLECTION_NAME = "AIC_2024_1"
DIMENSION = 1024 

# Set = True: Xóa sạch làm lại từ đầu
RESET_DB = True 

DATA_DIR = "./data_for_milvus" 

def import_data_hybrid():
    print(f"🔌 Connecting to Milvus at {MILVUS_URI}...")
    try:
        client = MilvusClient(uri=MILVUS_URI, db_name=DB_NAME)
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        print("💡 Gợi ý: Kiểm tra xem Docker Milvus đã chạy chưa? (docker ps)")
        return

    # --- BƯỚC 1: TẠO SCHEMA ---
    if RESET_DB:
        if client.has_collection(COLLECTION_NAME):
            print(f"🗑️ Dropping old collection...")
            client.drop_collection(COLLECTION_NAME)
        
        print("✨ Creating new schema with OCR support...")
        schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
        
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=DIMENSION)
        schema.add_field(field_name="video", datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(field_name="frame_id", datatype=DataType.INT64)
        schema.add_field(field_name="time", datatype=DataType.FLOAT)
        schema.add_field(field_name="path", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="ocr_text", datatype=DataType.VARCHAR, max_length=6000) 

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="embedding", 
            index_type="IVF_FLAT",
            metric_type="IP", 
            params={"nlist": 1024}
        )

        client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params
        )
        print(f"✅ Collection {COLLECTION_NAME} created successfully.")
    
    # --- BƯỚC 2: GHÉP DỮ LIỆU VÀ IMPORT ---
    parquet_files = glob.glob(os.path.join(DATA_DIR, "*.parquet"))
    print(f"📂 Found {len(parquet_files)} vector files.")

    total_inserted = 0
    
    for parquet_path in parquet_files:
        try:
            # 1. Đọc Vector
            df_vector = pd.read_parquet(parquet_path)
            
            # Lấy tên video
            video_name = os.path.basename(parquet_path).replace(".parquet", "")
            print(f"🔄 Processing {video_name}...", end=" ")
            
            # 2. Đọc OCR (Nếu có)
            ocr_path = os.path.join(DATA_DIR, f"{video_name}_ocr.csv")
            
            if os.path.exists(ocr_path):
                df_ocr = pd.read_csv(ocr_path)
                # Merge: Chỉ lấy cột ocr_text từ CSV ghép vào Parquet theo frame_id
                df_final = pd.merge(df_vector, df_ocr[['frame_id', 'ocr_text']], on='frame_id', how='left')
                
                # Xử lý dữ liệu rỗng
                df_final['ocr_text'] = df_final['ocr_text'].fillna("").astype(str)
            else:
                df_final = df_vector
                df_final['ocr_text'] = ""

            # 3. Đảm bảo kiểu dữ liệu (Tránh lỗi int vs float)
            df_final['frame_id'] = df_final['frame_id'].astype(int) 
            df_final['time'] = df_final['time'].astype(float)

            # 4. Insert
            data = df_final.to_dict('records')
            res = client.insert(collection_name=COLLECTION_NAME, data=data)
            
            count = res['insert_count']
            total_inserted += count
            print(f"✅ Inserted {count} frames.")
            
        except Exception as e:
            print(f"\n❌ Error processing {parquet_path}: {e}")

    print("="*40)
    print(f"🎉 DONE! Total vectors inserted: {total_inserted}")

if __name__ == "__main__":
    import_data_hybrid()
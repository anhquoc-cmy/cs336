import requests
import json

# URL của Backend Server (đang chạy ở Bước 1)
url = "http://localhost:8000/TextQuery"

# Nội dung tìm kiếm
payload = {
    "First_query": "a police", 
    "Next_query": "",
    "ocr_search": "" 
}

try:
    print(f"📡 Đang gửi request tới {url}...")
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Tìm thấy {data['total_results']} kết quả!")
        
        # In ra 3 kết quả đầu tiên để kiểm tra
        if data['total_results'] > 0:
            print("\nTop 3 kết quả:")
            for i, item in enumerate(data['kq'][:3]):
                entity = item['entity']
                print(f"{i+1}. Video: {entity.get('video')} | Frame: {entity.get('frame_id')} | Score: {item.get('distance'):.4f}")
                print(f"   Path: {entity.get('path')}")
    else:
        print("❌ Lỗi Server:", response.text)

except Exception as e:
    print(f"❌ Không kết nối được server (Server đã chạy chưa?): {e}")
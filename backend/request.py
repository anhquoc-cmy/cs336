import requests
import json
import sys

# Cấu hình Backend
BASE_URL = "http://localhost:8000"
ENDPOINT = "/TextQuery"

def send_query(first_query, next_query=""):
    """
    Gửi request tìm kiếm đến Backend FastAPI
    """
    url = f"{BASE_URL}{ENDPOINT}"
    
    payload = {
        "First_query": first_query,
        "Next_query": next_query
    }

    try:
        print(f"\n🚀 Đang gửi: '{first_query}'" + (f" -> '{next_query}'" if next_query else "") + " ...")
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        
        # Hiển thị kết quả
        print_result(data)

    except requests.exceptions.ConnectionError:
        print("❌ LỖI: Không thể kết nối đến Backend. Hãy chắc chắn server đang chạy ở http://localhost:8000")
    except requests.exceptions.HTTPError as err:
        print(f"❌ LỖI HTTP: {err}")
    except Exception as e:
        print(f"❌ LỖI KHÁC: {e}")

def print_result(data):
    """
    In TOP 10 kết quả trả về
    """
    print("=" * 65)
    total_found = data.get('total_results', 0)
    print(f"🔎 Tìm thấy tổng cộng: {total_found} kết quả (Hiển thị Top 10)")
    
    # Lấy danh sách kết quả
    results = data.get('kq')
    
    if not results:
        print("⚠️ Không tìm thấy kết quả nào.")
        print("=" * 65)
        return

    # Đảm bảo results luôn là list
    if not isinstance(results, list):
        results = [results]

    # --- CẮT LẤY TOP 10 ---
    top_results = results[:10]

    print("-" * 65)
    print(f"{'TOP':<4} | {'SCORE':<8} | {'VIDEO':<10} | {'FRAME':<8} | {'TIME':<8} | {'PATH'}")
    print("-" * 65)

    for index, item in enumerate(top_results):
        entity = item.get('entity', {})
        score = item.get('score', 0)
        
        video = entity.get('video', 'N/A')
        frame = entity.get('frame_id', 'N/A')
        time_sec = entity.get('time', 0)
        path = entity.get('path', '')
        
        # Làm tròn time cho gọn
        try:
            time_display = f"{float(time_sec):.2f}s"
        except:
            time_display = str(time_sec)

        print(f"#{index+1:<3} | {score:.4f}   | {video:<10} | {frame:<8} | {time_display:<8} | {path}")

    print("=" * 65)

def interactive_mode():
    print("\n--- INTERACTIVE SEARCH MODE (TOP 10) ---")
    print("Nhập 'q' hoặc 'exit' để thoát.")
    
    while True:
        try:
            q1 = input("\nNhập First Query: ").strip()
            if q1.lower() in ['q', 'exit']:
                break
            if not q1:
                continue

            q2 = input("Nhập Next Query (Enter để bỏ qua): ").strip()
            send_query(q1, q2)
            
        except KeyboardInterrupt:
            print("\nĐã thoát.")
            break

if __name__ == "__main__":
    if len(sys.argv) > 1:
        q1 = sys.argv[1]
        q2 = sys.argv[2] if len(sys.argv) > 2 else ""
        send_query(q1, q2)
    else:
        interactive_mode()
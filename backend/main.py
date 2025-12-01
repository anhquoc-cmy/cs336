"""
FastAPI Vector Search Service
===========================

A high-performance vector search service using CLIP models for image-text similarity search.
Supports temporal queries and provides both REST API and WebSocket interfaces.
"""
import numpy as np
import os
import json
import time
import logging
import asyncio
import unicodedata  
import re          

from typing import List, Optional, Dict, Any
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

# FastAPI imports
from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles  
from pydantic import BaseModel
from fastapi.responses import JSONResponse 
import requests
# ML/AI imports
import torch
import torch.nn.functional as F
import open_clip
from rapidfuzz import fuzz
# Vector database imports
from pymilvus import MilvusClient

# Configuration Management
# ========================

@dataclass
class ModelConfig:
    clip_model_name: str = "ViT-H-14-378-quickgelu"
    clip_pretrained: str = "dfn5b"
    device: str = "cuda" 

@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 19530
    database: str = "default"
    collection_name: str = "AIC_2024_1"
    search_limit: int = 3000
    replica_number: int = 1

@dataclass
class ServerConfig:
    cors_origins: str = "*" 
    max_workers: int = 4
    log_level: str = "INFO"
    gzip_minimum_size: int = 1000

class Config:
    def __init__(self, config_file: str = None):
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config_data = json.load(f)
        else:
            config_data = {}

        self.model = ModelConfig(
            clip_model_name=os.getenv("CLIP_MODEL_NAME", config_data.get("clip_model_name", "ViT-H-14-378-quickgelu")),
            clip_pretrained=os.getenv("CLIP_PRETRAINED", config_data.get("clip_pretrained", "dfn5b")),
            device=os.getenv("DEVICE", config_data.get("device", "cuda"))
        )

        self.database = DatabaseConfig(
            host=os.getenv("MILVUS_HOST", config_data.get("milvus_host", "localhost")),
            port=int(os.getenv("MILVUS_PORT", config_data.get("milvus_port", 19530))),
            database=os.getenv("MILVUS_DATABASE", config_data.get("milvus_database", "default")),
            collection_name=os.getenv("COLLECTION_NAME", config_data.get("collection_name", "AIC_2024_1")),
            search_limit=int(os.getenv("SEARCH_LIMIT", config_data.get("search_limit", 3000))),
            replica_number=int(os.getenv("REPLICA_NUMBER", config_data.get("replica_number", 1)))
        )

        self.server = ServerConfig(
            cors_origins=os.getenv("CORS_ORIGINS", config_data.get("cors_origins", "*")),
            max_workers=int(os.getenv("MAX_WORKERS", config_data.get("max_workers", 4))),
            log_level=os.getenv("LOG_LEVEL", config_data.get("log_level", "INFO")),
            gzip_minimum_size=int(os.getenv("GZIP_MIN_SIZE", config_data.get("gzip_minimum_size", 1000)))
        )

        if self.model.device == "cuda" and not torch.cuda.is_available():
            self.model.device = "cpu"
            logging.warning("CUDA requested but not available, falling back to CPU")

# Global Application State
# =======================
# --- CẤU HÌNH DRES SUBMISSION ---
DRES_BASE_URL = "http://192.168.28.151:5000"
USERNAME = "team013"  
PASSWORD = "123456" 

class DresSubmission(BaseModel):
    video_id: str
    timestamp_ms: int
    text_answer: Optional[str] = None
    
    
class VectorSearchService:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(self.config.model.device)

        logging.basicConfig(level=getattr(logging, self.config.server.log_level))
        self.logger = logging.getLogger(__name__)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.server.max_workers)

        self._initialize_models()
        self._initialize_database()
        self.active_connections: List[WebSocket] = []
        self.precomputed_tokens = {}

    def _initialize_models(self):
        self.logger.info("Initializing ML models...")
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            self.config.model.clip_model_name,
            pretrained=self.config.model.clip_pretrained
        )
        self.clip_model = self.clip_model.to(self.device)
        self.clip_tokenizer = open_clip.get_tokenizer(self.config.model.clip_model_name)
        self.logger.info("Models initialized successfully")

    def _initialize_database(self):
        self.logger.info("Initializing database connection...")
        try:
            self.milvus_client = MilvusClient(
                uri=f"http://{self.config.database.host}:{self.config.database.port}",
                db_name=self.config.database.database # Sửa db thành db_name cho đúng chuẩn
            )
            # Load collection để sẵn sàng search
            self.milvus_client.load_collection(
                collection_name=self.config.database.collection_name
            )
        except Exception as e:
            self.logger.error(f"Failed to connect or load collection: {e}")
            pass 
        self.logger.info("Database connection initialized successfully")

    @lru_cache(maxsize=1000)
    def encode_clip_text(self, query: str) -> torch.Tensor:
        text_inputs = self.clip_tokenizer([query]).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_inputs)
            return F.normalize(text_features, p=2, dim=-1)
            
    # --- HÀM MỚI: CLEAN TEXT ---
    def clean_text(self, text: str) -> str:
        """Chuyển tiếng Việt có dấu thành không dấu, xóa ký tự lạ"""
        if not text: return ""
        text = text.lower()
        text = unicodedata.normalize('NFD', text)
        text = "".join([c for c in text if unicodedata.category(c) != 'Mn'])
        text = text.replace('đ', 'd')
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # --- CẬP NHẬT: Nhận thêm tham số ocr_filter ---
    async def query_milvus(self, query_vector: torch.Tensor, ocr_filter: str = "", limit: int = None) -> List[Any]:
        if limit is None:
            limit = self.config.database.search_limit

        # 1. Tăng giới hạn tìm kiếm nếu có OCR (Over-fetching)
        search_limit = limit
        if ocr_filter:
            search_limit = limit * 2 # Lấy gấp đôi để lọc dần
        
        # 2. Tìm kiếm Vector (KHÔNG DÙNG filter="" ĐỂ TRÁNH LỖI)
        # Chúng ta bỏ tham số filter đi để Milvus hiểu là tìm trên toàn bộ dữ liệu
        res = await asyncio.to_thread(
            self.milvus_client.search,
            collection_name=self.config.database.collection_name,
            anns_field="embedding",
            data=[query_vector.tolist()[0]],
            limit=search_limit, # Dùng search_limit đã tính ở trên
            output_fields=['path', 'time', 'frame_id', 'video', 'ocr_text'],
            search_params={"metric_type": "IP", "params": {"nprobe": 128}} 
        )

        # 3. Chuyển đổi kết quả (Hit -> Dict) để tránh lỗi JSON Serializable
        clean_res = []
        for hits in res:
            clean_hits = []
            for hit in hits:
                hit_dict = hit.to_dict() if hasattr(hit, 'to_dict') else dict(hit)
                clean_hits.append(hit_dict)
            clean_res.append(clean_hits)
            
        results = clean_res[0] if clean_res else []

        # 4. Hậu xử lý: Fuzzy Matching OCR (Nếu người dùng có nhập OCR)
        if ocr_filter and len(results) > 0:
            filtered_results = []
            clean_query = self.clean_text(ocr_filter)
            
            for item in results:
                ocr_db = item['entity'].get('ocr_text', '')
                if not ocr_db: continue
                
                # So sánh độ tương đồng văn bản
                clean_db = self.clean_text(str(ocr_db))
                score = fuzz.partial_ratio(clean_query, clean_db)
                
                # Nếu giống trên 60%, giữ lại và cộng điểm
                if score >= 60:
                    bonus = (score / 100.0) * 0.15 
                    item['distance'] += bonus 
                    filtered_results.append(item)
            
            # Sắp xếp lại và cắt đúng số lượng yêu cầu
            filtered_results.sort(key=lambda x: x['distance'], reverse=True)
            return [filtered_results[:limit]]

        return [results[:limit]]

    # --- CẬP NHẬT: Nhận thêm ocr_query ---
    async def process_temporal_query(self, first_query: str, second_query: str = "", ocr_query: str = "") -> List[Any]:
        try:
            if second_query:
                first_encoded, second_encoded = await asyncio.gather(
                    asyncio.to_thread(self.encode_clip_text, first_query),
                    asyncio.to_thread(self.encode_clip_text, second_query)
                )
                
                # Chỉ áp dụng OCR filter cho câu đầu tiên (hoặc cả 2 tùy logic)
                fkq, nkq = await asyncio.gather(
                    self.query_milvus(first_encoded, ocr_filter=ocr_query), 
                    self.query_milvus(second_encoded) # Câu 2 thường là hành động, ít khi có chữ
                )
                
                list_1 = fkq[0] if fkq and len(fkq) > 0 else []
                list_2 = nkq[0] if nkq and len(nkq) > 0 else []
                result = self._process_temporal_relationships(list_1, list_2)
            else:
                first_encoded = await asyncio.to_thread(self.encode_clip_text, first_query)
                # Áp dụng OCR filter
                fkq = await self.query_milvus(first_encoded, ocr_filter=ocr_query)
                
                if fkq and len(fkq) > 0:
                    result = fkq[0]
                else:
                    result = []

            return result
        except Exception as e:
            self.logger.error(f"Error in temporal query processing: {e}")
            return []

    def _process_temporal_relationships(self, first_results: List[Any], second_results: List[Any]) -> List[Any]:
        if not first_results or not second_results:
            return []
            
        fkq_data = torch.tensor([
            [int(item['entity'].get('frame_id', 0)), item['distance'], hash(item['entity'].get('video', ''))]
            for item in first_results
        ], device=self.device)

        nkq_data = torch.tensor([
            [int(item['entity'].get('frame_id', 0)), item['distance'], hash(item['entity'].get('video', ''))]
            for item in second_results
        ], device=self.device)

        frame_diff = nkq_data[:, None, 0] - fkq_data[None, :, 0]
        same_video_mask = fkq_data[None, :, 2] == nkq_data[:, None, 2]
        valid_frame_diff_mask = (frame_diff > 0) & (frame_diff <= 1500) & same_video_mask

        score_increase = nkq_data[:, None, 1] * (1500 - frame_diff) / 1500
        score_increase = torch.where(valid_frame_diff_mask, score_increase, torch.zeros_like(score_increase))

        fkq_data[:, 1] += score_increase.max(dim=0).values
        scores = fkq_data[:, 1].detach().cpu().numpy() # Detach để an toàn

        sorted_indices = np.argsort(scores)[::-1][:1000]
        
        results = []
        for i in sorted_indices:
            item = first_results[i]
            item['entity']['debug_score_boost'] = float(scores[i] - item['distance'])
            item['distance'] = float(scores[i]) # Cập nhật lại score mới
            results.append(item)
            
        return results

# FastAPI Application Setup
# ========================

def create_app(config_file: str = None) -> FastAPI:
    config = Config(config_file)
    service = VectorSearchService(config)

    app = FastAPI(title="Vector Search Service")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(GZipMiddleware, minimum_size=config.server.gzip_minimum_size)

    # ĐƯỜNG DẪN ẢNH KEYFRAME
    PATH_TO_KEYFRAMES = r"C:\Users\USER\Desktop\output-keyframes" 
    if os.path.exists(PATH_TO_KEYFRAMES):
        app.mount("/static", StaticFiles(directory=PATH_TO_KEYFRAMES), name="static")
        print(f"✅ STATIC MOUNTED: {PATH_TO_KEYFRAMES}")
    else:
        print(f"⚠️ WARNING: Static folder not found at {PATH_TO_KEYFRAMES}")

    PATH_TO_VIDEOS = r"C:\Users\USER\Desktop\output-keyframes" 
    if os.path.exists(PATH_TO_VIDEOS):
        app.mount("/videos", StaticFiles(directory=PATH_TO_VIDEOS), name="videos")
        print(f"✅ VIDEO MOUNTED: {PATH_TO_VIDEOS}")
    else:
        print(f"⚠️ Cảnh báo: Không tìm thấy folder video tại {PATH_TO_VIDEOS}")


    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_json()
                
                # Lấy thông tin text OCR nếu có
                ocr_text = data.get('ocr_search', '')

                if data.get('type') == 'text_query':
                    result = await service.process_temporal_query(
                        data['firstQuery'], data.get('secondQuery', ''), ocr_query=ocr_text
                    )
                    await websocket.send_json({"kq": result})
                
                elif data.get('type') == 'multi_query':
                    first_q = ""
                    if 'queries' in data and len(data['queries']) > 0:
                        first_q = data['queries'][0].get('content', '')
                    elif 'temporal' in data and len(data['temporal']) > 0:
                        first_q = data['temporal'][0].get('content', '')
                    
                    if first_q:
                        result = await service.process_temporal_query(first_q, "", ocr_query="")
                        await websocket.send_json({"kq": result})

        except Exception as e:
            print(f"WS Error: {e}")

    async def dummy_socket(websocket: WebSocket):
        await websocket.accept()
        try:
            while True: await websocket.receive_text()
        except: pass

    app.websocket("/ws/share_image")(dummy_socket)
    app.websocket("/ws/log")(dummy_socket)
    app.websocket("/ws/group_search")(dummy_socket)
    app.websocket("/ws/similarity_search")(dummy_socket)
    
    # --- SOCKET LỌC CÓ THỂ DÙNG SAU NÀY ---
    @app.websocket("/ws/filter_query")
    async def ws_filter(websocket: WebSocket):
        await websocket.accept()
        try:
            while True: 
                # Nếu frontend gửi filter qua đây, có thể xử lý sau
                await websocket.receive_text()
        except: pass

    app.websocket("/ws/share_query")(dummy_socket)
    app.websocket("/ws/alerts")(dummy_socket)
    app.websocket("/ws/pagnition")(dummy_socket)

    @app.post("/TextQuery")
    async def text_query_endpoint(request: Request):
        try:
            body = await request.json()
            first_query = body.get('First_query', '')
            next_query = body.get('Next_query', '')
            
            # Lấy text OCR từ request
            ocr_text = body.get('ocr_search', '') 

            result = await service.process_temporal_query(first_query, next_query, ocr_query=ocr_text)

            return {
                "kq": result,
                "fquery": first_query,
                "nquery": next_query,
                "total_results": len(result)
            }

        except Exception as e:
            service.logger.error(f"Error in text query: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}
    # --- API MỚI: XỬ LÝ SUBMIT DRES ---
    @app.post("/dres/submit")
    async def proxy_submit_to_dres(submission: DresSubmission):
        try:
            # 1. Login để lấy Session ID
            login_url = f"{DRES_BASE_URL}/api/v2/login"
            login_payload = {"username": USERNAME, "password": PASSWORD}
            
            session_resp = requests.post(login_url, json=login_payload, timeout=5)
            if not session_resp.ok:
                raise HTTPException(status_code=400, detail=f"DRES Login failed: {session_resp.text}")
                
            session_data = session_resp.json()
            # Lấy sessionId, tùy version DRES key có thể là sessionId hoặc session_id
            session_id = session_data.get("sessionId") or session_data.get("sessionID")

            # 2. Lấy Evaluation ID đang Active
            eval_url = f"{DRES_BASE_URL}/api/v2/client/evaluation/list"
            eval_resp = requests.get(eval_url, params={"session": session_id}, timeout=5)
            if not eval_resp.ok:
                 raise HTTPException(status_code=400, detail="Failed to list evaluations")
            
            evaluations = eval_resp.json()
            active_eval = next((e for e in evaluations if str(e.get("status")).upper() == "ACTIVE"), None)
            
            if not active_eval:
                raise HTTPException(status_code=404, detail="No ACTIVE evaluation found")
            
            evaluation_id = active_eval.get("id")

            # 3. Gửi Submit
            submit_url = f"{DRES_BASE_URL}/api/v2/submit/{evaluation_id}"
            
            # Tính toán thời gian (ms)
            timestamp_ms = submission.timestamp_ms
            
            if submission.text_answer:
                # Logic cho VQA
                answer_text = f"QA-{submission.text_answer}-{submission.video_id}-{timestamp_ms}"
                body = {"answerSets": [{"answers": [{"text": answer_text}]}]}
            else:
                # Logic cho KIS
                body = {
                    "answerSets": [{
                        "answers": [{
                            "mediaItemName": submission.video_id,
                            "start": timestamp_ms,
                            "end": timestamp_ms
                        }]
                    }]
                }

            final_resp = requests.post(submit_url, params={"session": session_id}, json=body, timeout=10)
            
            if not final_resp.ok:
                 return JSONResponse(status_code=final_resp.status_code, content=final_resp.json())

            return final_resp.json()

        except Exception as e:
            print(f"Error submitting to DRES: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    return app

if __name__ == "__main__":
    import uvicorn
    config_file = os.getenv("CONFIG_FILE", "config.json")
    app = create_app(config_file)
    
    print("🚀 Server starting on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
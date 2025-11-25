import os
import glob
import torch
import logging
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import io
import time

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="DeepSeek-OCR Service")

# --- Configuration ---
MODEL_NAME = "deepseek-ai/DeepSeek-OCR"  # Will download on first run
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Initializing DeepSeek-OCR on {DEVICE}...")

# --- Load Model (Global) ---
# Note: Actual loading code depends on specific repo structure. 
# Assuming HuggingFace AutoModel support or using the official wrapper.
# If strictly custom, we would clone the repo in Dockerfile. 
# Below is the standard HuggingFace pattern for DeepSeek-VL/OCR.

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_NAME, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)
    model.eval()
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Model load failed: {e}")
    # Placeholder for build process
    model = None
    tokenizer = None

class FolderRequest(BaseModel):
    folder_path: str
    output_format: str = "markdown"  # markdown or text

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Method={request.method} Path={request.url.path} Status={response.status_code} Time={process_time:.2f}s")
    return response

@app.get("/health")
def health():
    return {"status": "running", "device": DEVICE, "model": MODEL_NAME}

@app.post("/scan_folder")
async def scan_folder_api(request: FolderRequest):
    """
    Scans a mounted directory for images and generates .md files.
    """
    logger.info(f"Received scan request for folder: {request.folder_path}")
    
    if not os.path.exists(request.folder_path):
        logger.error(f"Path not found: {request.folder_path}")
        raise HTTPException(status_code=404, detail=f"Path not found: {request.folder_path}")

    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.webp']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(request.folder_path, ext)))
    
    logger.info(f"Found {len(files)} images to process.")

    results = []

    for img_path in files:
        try:
            # Check if output already exists to skip
            out_path = os.path.splitext(img_path)[0] + ".md"
            if os.path.exists(out_path):
                logger.info(f"Skipping {img_path} (already processed)")
                results.append({"file": img_path, "status": "skipped"})
                continue

            logger.info(f"Processing {img_path}...")
            
            # --- Inference Logic ---
            # This is the "Context Optical Compression" usage
            # 2. Generate using model.infer
            # model.infer(tokenizer, prompt, image_file, output_path, save_results=True)
            # It saves the result directly to the output_path (which is a directory)
            
            model.infer(
                tokenizer=tokenizer,
                prompt="Convert this page to markdown.",
                image_file=img_path,
                output_path=request.folder_path, # Save in the same folder
                save_results=True
            )
            
            # Since model.infer returns None and saves to file, we verify the file exists
            if os.path.exists(out_path):
                logger.info(f"Successfully generated {out_path}")
                results.append({"file": img_path, "status": "processed"})
            else:
                logger.warning(f"Output file not found for {img_path}")
                results.append({"file": img_path, "status": "processed_but_file_missing"})
            
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            results.append({"file": img_path, "status": "error", "error": str(e)})

    summary = {"total_processed": len(results), "details": results}
    logger.info(f"Scan complete. Summary: {summary}")
    return summary
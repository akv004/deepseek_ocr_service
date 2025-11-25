import os
import glob
import torch
import logging
import sys
import shutil
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer
from PIL import Image

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("DeepSeek-OCR")

app = FastAPI(title="DeepSeek-OCR Service")

# --- CONFIG ---
# Ensure this matches the HuggingFace ID exactly
MODEL_NAME = "deepseek-ai/DeepSeek-OCR"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"🚀 Initializing on {DEVICE}...")

# --- LOAD MODEL (Global) ---
try:
    # We force use_fast=False to avoid Rust tokenizer issues
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_fast=False,
    )

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        attn_implementation="eager"
    ).to(DEVICE)
    model.eval()
    logger.info("✅ Model loaded successfully.")
except Exception as e:
    logger.critical(f"❌ FATAL: Model load failed: {e}")
    model = None

# --- REQUEST MODELS ---
class FolderRequest(BaseModel):
    folder_path: str


class ImageRequest(BaseModel):
    image_path: str


# --- CORE INFERENCE LOGIC ---
def run_deepseek_inference(image_path: str):
    """
    Uses the correct .infer() method from DeepSeek-OCR's custom code.
    """
    logger.info(f"📂 Reading image file: {image_path}")

    # Safety check for the method existence
    if not hasattr(model, 'infer'):
        # Fallback for models that might use standard generation (rare for this specific repo)
        logger.warning(f"Model object {type(model)} has no .infer(). Trying standard chat...")
        if hasattr(model, 'chat'):
            return model.chat(tokenizer, Image.open(image_path).convert('RGB'), "Convert to markdown")
        raise RuntimeError("Model does not support .infer() or .chat().")

    # Setup temp path for the model to write its sidecar file
    temp_dir = os.path.dirname(image_path)
    expected_md_file = os.path.splitext(image_path)[0] + ".md"

    # model.infer() writes to disk. We let it write, then read the file back.
    # But with crop_mode=False, it might not write to file.
    # So we try model.chat first if crop_mode is False.
    
    # Attempt 1: Try model.chat (Standard HF API)
    if hasattr(model, "chat"):
        try:
            logger.info("🤖 Attempting model.chat()...")
            # model.chat usually returns the generated text directly
            response = model.chat(
                tokenizer=tokenizer,
                image=Image.open(image_path).convert("RGB"),
                prompt="Convert this page to markdown."
            )
            
            # Check if response is valid text
            if response and isinstance(response, str):
                logger.info(f"✅ model.chat returned {len(response)} characters.")
                logger.info(f"💾 Writing output to: {expected_md_file}")
                with open(expected_md_file, "w", encoding="utf-8") as f_out:
                    f_out.write(response)
                return response
            else:
                logger.warning(f"model.chat returned unexpected type: {type(response)}")
                
        except Exception as e:
            logger.warning(f"model.chat failed: {e}. Falling back to model.infer")

    # Attempt 2: Try model.infer (Custom DeepSeek API)
    logger.info("🤖 Attempting model.infer()...")
    try:
        import io
        import contextlib
        
        # Capture stdout because model.infer prints the result instead of returning it
        f_capture = io.StringIO()
        with contextlib.redirect_stdout(f_capture):
            model.infer(
                tokenizer=tokenizer,
                prompt="Convert this page to markdown.",
                image_file=image_path,
                output_path=temp_dir,
                base_size=1024,
                image_size=1024,
                crop_mode=False,  # Disable Gundam mode to avoid CUBLAS errors
                save_results=True  # Required to generate the file (though it seems to fail, we capture stdout)
            )
        
        captured_output = f_capture.getvalue()
        
        # Filter out known noise logs if necessary, or just save everything.
        # The logs show some headers like "BASE: ...", "NO PATCHES", "=====". 
        # We might want to keep it simple for now and just save it.
        
        if captured_output:
            logger.info(f"✅ Captured {len(captured_output)} chars from stdout.")
            
            # Clean up the output (remove the "=====" and "BASE:" debug info if possible)
            # Simple heuristic: The actual content usually starts after the last "====="
            if "=====" in captured_output:
                # Find the last occurrence of "=====" and take everything after it
                clean_content = captured_output.split("=====")[-1].strip()
            else:
                clean_content = captured_output
            
            logger.info(f"💾 Writing captured content to: {expected_md_file}")
            with open(expected_md_file, "w", encoding="utf-8") as f_out:
                f_out.write(clean_content)
            return clean_content
        else:
            logger.warning("⚠️ model.infer produced no stdout output.")
            
    except Exception as e:
        logger.error(f"❌ model.infer failed: {e}")

    # Final Check: Did the file get created?
    if os.path.exists(expected_md_file):
        logger.info(f"✅ Verified generated file exists at: {expected_md_file}")
        with open(expected_md_file, "r", encoding="utf-8") as f:
            return f.read()
    else:
        logger.error(f"❌ Failed to generate file at: {expected_md_file}")
        return "Error: Markdown file was not generated by model."


# --- ENDPOINT 1: HEALTH CHECK (Fixing your 404) ---
@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model failed to load")
    return {"status": "running", "device": DEVICE, "model": MODEL_NAME}


# --- ENDPOINT 2: IMMEDIATE RESPONSE ---
@app.post("/read_image")
async def read_image_api(request: ImageRequest):
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail=f"Image not found at {request.image_path}")

    logger.info(f"📖 Reading single image: {request.image_path}")

    try:
        markdown_text = run_deepseek_inference(request.image_path)
        return {
            "file": request.image_path,
            "content": markdown_text
        }
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- ENDPOINT 3: BATCH PROCESSING ---
def process_folder_task(folder_path: str):
    logger.info(f"📂 STARTING BATCH SCAN: {folder_path}")
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.webp']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder_path, ext)))

    for img_path in files:
        try:
            out_path = os.path.splitext(img_path)[0] + ".md"
            if os.path.exists(out_path): continue

            logger.info(f"Processing: {os.path.basename(img_path)}")
            run_deepseek_inference(img_path)

        except Exception as e:
            logger.error(f"Failed on {img_path}: {e}")

    logger.info("🏁 Batch scan complete.")


@app.post("/scan_folder")
async def scan_folder_api(request: FolderRequest, background_tasks: BackgroundTasks):
    if not os.path.exists(request.folder_path):
        raise HTTPException(status_code=404, detail="Folder not found")

    background_tasks.add_task(process_folder_task, request.folder_path)
    return {"status": "Job Queued", "mode": "batch_save_to_disk"}
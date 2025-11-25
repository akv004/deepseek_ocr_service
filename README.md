# DeepSeek OCR Service

A Dockerized service for DeepSeek-OCR, optimized for NVIDIA RTX 5090.

## Prerequisites

- **NVIDIA GPU**: RTX 5090 (or other Ampere/Hopper/Blackwell GPU).
- **Docker**: With NVIDIA Container Toolkit installed and configured.
- **Hugging Face Token**: You must have `HUGGINGFACE_HUB_TOKEN` exported in your host environment (e.g., `.zshrc`).

## Local Verification (Optional)

Before running in Docker, you can verify that the local Conda environment works correctly on your host.

1.  Activate the environment:
    ```bash
    conda activate deepseek-ocr
    ```
2.  Run the test script:
    ```bash
    python test_local.py
    ```
    *If this prints "SUCCESS", your local environment is correct.*

## Setup

The service uses a **multi-stage Docker build** to create the environment and compile `flash-attn` inside the container. This ensures binary compatibility with the container's OS.

**Note:** You do **not** need to create `deepseek_env.tar.gz` on your host. The Docker build process handles everything.

## Running the Service

1.  **Start the Service**:
    ```bash
    docker compose up --build
    ```
    *The `--build` flag is recommended for the first run to ensure the image is created from the packed environment.*

2.  **Access the API**:
    The service runs on port `8004`.
    - Health Check: `http://localhost:8004/health`
    - Docs: `http://localhost:8004/docs`

## How to Work with This Service

After the first build, you can start the service in detached mode:
```bash
docker compose up -d
```
*No `--build` flag is needed after the initial build.*

To stop the service, run:
```bash
docker compose down
```
This will gracefully shut down and remove the running containers.

To make changes to the API, simply edit `app.py` locally. The container runs `uvicorn` with the `--reload` flag, so any changes to `app.py` will be detected and the API will restart automatically inside the container. This provides zero wait time for code updates—no need to rebuild or restart the container manually.

## Testing

You can test the service using `curl` or any API client (like Postman).

### 1. Health Check
Verify the service is running and the model is loaded.
```bash
curl -X GET "http://localhost:8004/health"
```
**Expected Response:**
```json
{"status":"running", "device":"cuda", "model":"deepseek-ai/DeepSeek-OCR"}
```

### 2. Scan Folder
Process images in the `capture` directory.
1.  Place some images (jpg, png) in the `capture/` folder on your host.
2.  Run the scan command:
```bash
curl -X POST "http://localhost:8004/scan_folder" \
     -H "Content-Type: application/json" \
     -d '{"folder_path": "/app/capture"}'
```
*Note: The `folder_path` must be `/app/capture` because that is where the volume is mounted inside the Docker container.*

## Configuration

- **Hugging Face Cache**: The service mounts your host's global cache (`~/projects/.cache/huggingface`) to avoid re-downloading models.
- **Input Directory**: The service scans the local `./capture` directory. Place images here to be processed.

## Directory Structure
```
.
├── app.py
├── capture/          # Place images here
├── deepseek_env.tar.gz
├── deepseek_ocr_env.yml
├── docker-compose.yml
├── Dockerfile
├── install.txt
└── README.md
```

- **Flash Attention**: If you see warnings about flash attention, ensure you are using the packed environment (`deepseek_env.tar.gz` exists in the directory).
- **Permissions**: If you encounter permission issues with the cache, ensure your user ID matches the container (default root, but volume mapping usually handles this).




docker compose restart deepseek-ocr
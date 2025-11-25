import requests
import json
import time
import os
import sys

# --- CONFIGURATION ---
BASE_URL = "http://localhost:8004"
# IMPORTANT: This is the path INSIDE the Docker container
CONTAINER_CAPTURE_PATH = "/app/capture"
# This is the path on your HOST machine (for verifying files exist)
LOCAL_CAPTURE_PATH = "./capture"


# --- COLORS FOR TERMINAL ---
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_status(message, status="INFO"):
    if status == "INFO":
        print(f"{Colors.OKCYAN}[INFO]{Colors.ENDC} {message}")
    elif status == "SUCCESS":
        print(f"{Colors.OKGREEN}[SUCCESS]{Colors.ENDC} {message}")
    elif status == "ERROR":
        print(f"{Colors.FAIL}[ERROR]{Colors.ENDC} {message}")
    elif status == "WARN":
        print(f"{Colors.WARNING}[WARN]{Colors.ENDC} {message}")


def check_health():
    print(f"\n{Colors.HEADER}--- 1. Checking Health ---{Colors.ENDC}")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print_status(f"Service is Online! {response.json()}", "SUCCESS")
            return True
        else:
            print_status(f"Service returned {response.status_code}: {response.text}", "ERROR")
            return False
    except requests.exceptions.ConnectionError:
        print_status("Could not connect to localhost:8004. Is Docker running?", "ERROR")
        return False


def test_single_image():
    print(f"\n{Colors.HEADER}--- 2. Testing Single Image (Synchronous) ---{Colors.ENDC}")

    # 1. Find a real image to test
    valid_exts = ('.png', '.jpg', '.jpeg', '.webp')
    images = [f for f in os.listdir(LOCAL_CAPTURE_PATH) if f.lower().endswith(valid_exts)]

    if not images:
        print_status(f"No images found in {LOCAL_CAPTURE_PATH}. Please add a test image.", "WARN")
        return

    test_image = images[0]
    # Map local filename to container path
    container_path = f"{CONTAINER_CAPTURE_PATH}/{test_image}"

    print_status(f"Testing with image: {test_image}")
    print_status(f"Sending Path (Docker): {container_path}")

    start_time = time.time()
    payload = {"image_path": container_path}

    try:
        response = requests.post(f"{BASE_URL}/read_image", json=payload)
        duration = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            content_preview = data['content'][:100].replace('\n', ' ') + "..."
            print_status(f"Response received in {duration:.2f}s", "SUCCESS")
            print(f"{Colors.OKBLUE}Markdown Output (Preview):{Colors.ENDC} {content_preview}")
        else:
            print_status(f"Failed: {response.text}", "ERROR")

    except Exception as e:
        print_status(f"Request Error: {e}", "ERROR")


def test_batch_scan():
    print(f"\n{Colors.HEADER}--- 3. Testing Batch Scan (Background) ---{Colors.ENDC}")

    payload = {"folder_path": CONTAINER_CAPTURE_PATH}

    try:
        response = requests.post(f"{BASE_URL}/scan_folder", json=payload)

        if response.status_code == 200:
            print_status("Batch Job Triggered Successfully", "SUCCESS")
            print(f"Server Response: {response.json()}")
            print_status(f"Check your local '{LOCAL_CAPTURE_PATH}' folder. .md files should appear shortly.", "INFO")
        else:
            print_status(f"Failed to trigger batch: {response.text}", "ERROR")

    except Exception as e:
        print_status(f"Request Error: {e}", "ERROR")


if __name__ == "__main__":
    if not os.path.exists(LOCAL_CAPTURE_PATH):
        print_status(f"Creating local capture folder: {LOCAL_CAPTURE_PATH}", "WARN")
        os.makedirs(LOCAL_CAPTURE_PATH)

    if check_health():
        test_single_image()
        test_batch_scan()
    else:
        sys.exit(1)
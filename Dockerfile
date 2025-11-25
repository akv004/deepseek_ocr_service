# Use NVIDIA CUDA base for Ubuntu 24.04 (Matches Host OS for binary compatibility)
# Using CUDA 13.0.2 because the host environment was compiled against CUDA 13 (libcudart.so.13)
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Set path and library path
ENV PATH="/opt/deepseek-ocr/bin:$PATH"
ENV LD_LIBRARY_PATH="/opt/deepseek-ocr/lib:$LD_LIBRARY_PATH"

# Copy and unpack the pre-built environment from the host
# Since Host and Container are both Ubuntu 24.04, this works perfectly!
COPY deepseek_env.tar.gz /app/deepseek_env.tar.gz
RUN mkdir -p /opt/deepseek-ocr && \
    tar -xzf deepseek_env.tar.gz -C /opt/deepseek-ocr && \
    rm deepseek_env.tar.gz && \
    conda-unpack && \
    pip install transformers==4.46.3 accelerate tokenizers matplotlib

# Copy Application Code
COPY app.py .

# Expose API port
EXPOSE 8004

# Run the API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8004"]
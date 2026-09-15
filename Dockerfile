# RunPod Serverless GPU worker for moncusoai/wvl81
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        runpod \
        "transformers>=4.51.0" \
        accelerate \
        safetensors \
        huggingface-hub \
        tokenizers \
        einops \
        "qwen-vl-utils>=0.0.8" \
        pillow \
        sentencepiece \
        protobuf

COPY handler.py /app/handler.py

ENV MODEL_PATH=/runpod-volume/models/moncusoai/wvl81
ENV PRELOAD_MODEL=1
ENV HF_HOME=/runpod-volume/hf-cache
ENV TRANSFORMERS_CACHE=/runpod-volume/hf-cache

CMD ["python", "-u", "handler.py"]

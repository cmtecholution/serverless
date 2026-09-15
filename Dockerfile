# RunPod Serverless GPU worker for moncusoai/wvl81
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /

# Use the image Python that already has CUDA torch; add torchvision for Qwen3-VL processor
RUN python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        "torchvision==0.19.1" \
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
        protobuf && \
    python -c "import torch, torchvision; from transformers import AutoProcessor; print('ok', torch.__version__, torchvision.__version__)"

COPY rp_handler.py /

ENV MODEL_PATH=/runpod-volume/myapp/models/moncusoai/wvl81
ENV PRELOAD_MODEL=0
ENV HF_HOME=/runpod-volume/hf-cache
ENV TRANSFORMERS_CACHE=/runpod-volume/hf-cache
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "/rp_handler.py"]

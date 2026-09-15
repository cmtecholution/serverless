# Serverless worker: use volume Python 3.12 + moncusoai venv site-packages.
# The venv's bin/python is a symlink to /usr/bin/python3.12 (missing in the
# Serverless image → exit 127). Use the portable interpreter on the volume instead.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /

COPY rp_handler.py /rp_handler.py
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Source on network volume; start.sh stages to LOCAL then preloads into VRAM.
ENV MODEL_PATH=/runpod-volume/myapp/models/moncusoai/wvl81
ENV LOCAL_MODEL_PATH=/local/models/wvl81
ENV STAGE_MODEL=1
ENV PRELOAD_MODEL=1
ENV HF_HOME=/runpod-volume/hf-cache
ENV TRANSFORMERS_CACHE=/runpod-volume/hf-cache
ENV PYTHONUNBUFFERED=1
ENV PYTHONNOUSERSITE=1
ENV VIRTUAL_ENV=/runpod-volume/venvs/moncusoai
ENV PYTHONPATH=/runpod-volume/venvs/moncusoai/lib/python3.12/site-packages
ENV PATH=/runpod-volume/venvs/moncusoai/bin:/runpod-volume/uv-python/cpython-3.12.14-linux-x86_64-gnu/bin:$PATH

CMD ["/start.sh"]

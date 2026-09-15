# Thin Serverless wrapper: inference uses the persistent volume venv
# (/workspace/venvs/moncusoai on pods == /runpod-volume/venvs/moncusoai on Serverless)
#
# Update GitHub (cmtecholution/serverless) with this Dockerfile + rp_handler.py, then
# redeploy/rebuild the endpoint. Network volume must stay attached.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /

COPY rp_handler.py /rp_handler.py

ENV MODEL_PATH=/runpod-volume/myapp/models/moncusoai/wvl81
ENV PRELOAD_MODEL=0
ENV HF_HOME=/runpod-volume/hf-cache
ENV TRANSFORMERS_CACHE=/runpod-volume/hf-cache
ENV PYTHONUNBUFFERED=1
ENV PYTHONNOUSERSITE=1

CMD ["/runpod-volume/venvs/moncusoai/bin/python", "-u", "/rp_handler.py"]

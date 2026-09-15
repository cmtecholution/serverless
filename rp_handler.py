# =============================================================================
# RunPod Serverless worker – wvl81 (Qwen3-VL) streaming inference
# =============================================================================
# Network volume path on Serverless is /runpod-volume (same data as /workspace on pods)
# =============================================================================

import os
import threading
from typing import Dict, Generator, List

import runpod
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, TextIteratorStreamer

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/runpod-volume/myapp/models/moncusoai/wvl81",
)

_model = None
_processor = None
_lock = threading.Lock()


def load_model():
    global _model, _processor
    if _model is not None:
        return

    print(f"Loading Qwen3-VL from {MODEL_PATH}")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else "cpu"

    _processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    _model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa" if torch.cuda.is_available() else None,
    )
    _model.eval()
    print("Model loaded")


def _prepare_inputs(messages: List[Dict]):
    from qwen_vl_utils import process_vision_info

    text = _processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = _processor(
        text=[text],
        images=image_inputs if image_inputs else None,
        videos=video_inputs if video_inputs else None,
        padding=True,
        return_tensors="pt",
    )
    if torch.cuda.is_available():
        inputs = inputs.to(_model.device)
    return inputs


def generate_tokens(job_input: dict) -> Generator[dict, None, None]:
    messages = job_input.get("messages")
    if not messages:
        yield {"error": "input.messages is required"}
        return

    max_new_tokens = int(job_input.get("max_new_tokens", 2048))
    temperature = float(job_input.get("temperature", 0.7))
    top_p = float(job_input.get("top_p", 0.9))
    repetition_penalty = float(job_input.get("repetition_penalty", 1.05))

    with _lock:
        load_model()
        inputs = _prepare_inputs(messages)
        tokenizer = getattr(_processor, "tokenizer", _processor)
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            streamer=streamer,
        )
        thread = threading.Thread(target=_model.generate, kwargs=gen_kwargs)
        thread.start()
        for token in streamer:
            if token:
                yield {"token": token}
        thread.join(timeout=3600)


def handler(event):
    """Streaming handler – each yield is a token chunk for /stream."""
    job_input = event.get("input") or {}
    yield from generate_tokens(job_input)


if __name__ == "__main__":
    if os.environ.get("PRELOAD_MODEL", "1") == "1":
        try:
            load_model()
        except Exception as e:
            print(f"Preload skipped/failed (will retry on first job): {e}")

    runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})

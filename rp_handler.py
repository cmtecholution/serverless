import os
import sys
import threading
import time
from typing import Dict, Generator, List

print(f"Python: {sys.executable}", flush=True)
print(f"sys.path[0:3]={sys.path[:3]}", flush=True)

import runpod
import torch
import torchvision  # required by Qwen3VLVideoProcessor
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, TextIteratorStreamer

print(
    f"torch={torch.__version__} torchvision={torchvision.__version__} "
    f"cuda={torch.cuda.is_available()}",
    flush=True,
)

# Prefer locally staged weights (set by start.sh). Fall back to volume path.
MODEL_PATH = os.environ.get("MODEL_LOAD_PATH") or os.environ.get(
    "LOCAL_MODEL_PATH"
) or os.environ.get(
    "MODEL_PATH",
    "/runpod-volume/myapp/models/moncusoai/wvl81",
)

_PRELOAD_DEFAULT = "1"

_model = None
_processor = None
_lock = threading.Lock()


def _truthy(val: str) -> bool:
    return val.strip().lower() in ("1", "true", "yes", "on")


def load_model():
    """Load once into GPU RAM. Subsequent calls are no-ops."""
    global _model, _processor
    if _model is not None:
        return

    t0 = time.perf_counter()
    print(f"Loading Qwen3-VL from {MODEL_PATH}", flush=True)
    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(f"MODEL_PATH does not exist: {MODEL_PATH}")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else "cpu"

    _processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )
    _model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="sdpa" if torch.cuda.is_available() else None,
    )
    _model.eval()

    if torch.cuda.is_available():
        try:
            torch.zeros(1, device=next(_model.parameters()).device)
            torch.cuda.synchronize()
        except Exception as e:
            print(f"CUDA warmup skipped: {e}", flush=True)

    print(f"Model loaded in {time.perf_counter() - t0:.1f}s from {MODEL_PATH}", flush=True)


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
        inputs = inputs.to(next(_model.parameters()).device)
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
        if _model is None:
            print("WARNING: model not preloaded — cold load on request path", flush=True)
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
    print("Worker Start", flush=True)
    job_input = event.get("input") or {}
    try:
        yield from generate_tokens(job_input)
    except Exception as e:
        print(f"Handler error: {e}", flush=True)
        yield {"error": str(e)}
        raise


if __name__ == "__main__":
    print(
        f"Boot config: MODEL_LOAD_PATH={MODEL_PATH} "
        f"PRELOAD={os.environ.get('PRELOAD_MODEL', _PRELOAD_DEFAULT)} "
        f"STAGE={os.environ.get('STAGE_MODEL', '1')}",
        flush=True,
    )
    if _truthy(os.environ.get("PRELOAD_MODEL", _PRELOAD_DEFAULT)):
        print(
            "PRELOAD_MODEL enabled — loading into GPU before accepting jobs...",
            flush=True,
        )
        load_model()
    else:
        print(
            "PRELOAD_MODEL disabled — first job will pay full weight load",
            flush=True,
        )

    runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})

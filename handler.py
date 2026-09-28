import os
import sys
import threading
import time
import traceback
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

    # Log a compact shape of the request for debugging hung jobs
    try:
        summary = []
        for m in messages:
            kinds = []
            c = m.get("content")
            if isinstance(c, str):
                kinds.append(f"text:{len(c)}")
            elif isinstance(c, list):
                for p in c:
                    if isinstance(p, dict):
                        kinds.append(p.get("type", "?"))
            summary.append(f"{m.get('role')}:[{','.join(kinds)}]")
        print("prepare_inputs " + " | ".join(summary), flush=True)
    except Exception:
        pass

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


def _close_streamer(streamer: TextIteratorStreamer) -> None:
    """Unblock `for token in streamer` after a failed generate() thread."""
    try:
        streamer.end()
        return
    except Exception:
        pass
    try:
        stop = getattr(streamer, "stop_signal", None)
        q = getattr(streamer, "text_queue", None)
        if q is not None:
            q.put(stop)
    except Exception:
        pass


def generate_tokens(job_input: dict) -> Generator[dict, None, None]:
    messages = job_input.get("messages")
    if not messages:
        yield {"error": "input.messages is required"}
        return

    max_new_tokens = int(job_input.get("max_new_tokens", 2048))
    max_new_tokens = max(1, min(max_new_tokens, 4096))
    temperature = float(job_input.get("temperature", 0.7))
    top_p = float(job_input.get("top_p", 0.9))
    repetition_penalty = float(job_input.get("repetition_penalty", 1.05))

    # temperature==0 + do_sample=True crashes / hangs some torch builds
    # (matches Encompass analyze which sends temperature: 0).
    do_sample = temperature > 1e-5

    with _lock:
        if _model is None:
            print("WARNING: model not preloaded — cold load on request path", flush=True)
        load_model()
        try:
            inputs = _prepare_inputs(messages)
        except Exception as e:
            traceback.print_exc()
            yield {"error": f"prepare_inputs failed: {e}"}
            return

        tokenizer = getattr(_processor, "tokenizer", _processor)
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            streamer=streamer,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            gen_kwargs["repetition_penalty"] = repetition_penalty

        err_box: List[BaseException] = []

        def _run_generate() -> None:
            try:
                print(
                    f"generate start do_sample={do_sample} "
                    f"max_new_tokens={max_new_tokens} temperature={temperature}",
                    flush=True,
                )
                _model.generate(**gen_kwargs)
                print("generate done", flush=True)
            except Exception as e:
                err_box.append(e)
                print(f"generate FAILED: {e}", flush=True)
                traceback.print_exc()
                _close_streamer(streamer)

        thread = threading.Thread(target=_run_generate, name="generate", daemon=True)
        thread.start()
        try:
            for token in streamer:
                if token:
                    yield {"token": token}
        finally:
            thread.join(timeout=60)

        if err_box:
            yield {"error": f"generate failed: {err_box[0]}"}
            return
        if thread.is_alive():
            yield {"error": "generate thread still running after stream end"}
            return


def handler(event):
    print("Worker Start", flush=True)
    job_input = event.get("input") or {}
    try:
        yielded = 0
        for item in generate_tokens(job_input):
            yielded += 1
            yield item
        if yielded == 0:
            yield {"error": "model produced no tokens"}
    except Exception as e:
        print(f"Handler error: {e}", flush=True)
        traceback.print_exc()
        yield {"error": str(e)}


# Start the Serverless function when the script is run (RunPod Hub detects this).
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

    # Required by RunPod queue workers / Hub repo scanner:
    runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})

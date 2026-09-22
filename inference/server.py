"""Local, single-model inference service. Run with python -m inference.server."""

import argparse
import base64
import binascii
import io
import logging
import math
import threading
import time
import warnings
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from jev_navigation.policy import ACTIONS, OPTIONS, QUESTION, validate_probabilities

MODEL_ID = "Mapika/decider-2b-vision"
MODEL_REVISION = "446c8c5e334e53ae3526a1c5384f93d5caee68cd"
MAX_BODY = 3_000_000
Image.MAX_IMAGE_PIXELS = 4_000_000


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    request_id: int = Field(ge=0)
    image_base64: str = Field(min_length=4, max_length=2_800_000)
    goal: str = Field(min_length=1, max_length=512)
    previous_action: Literal["forward", "gentle_left", "gentle_right", "left", "right",
                             "stop", "goal_reached"] = "stop"


def decode_image(encoded):
    try:
        raw = base64.b64decode(encoded, validate=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as image:
                if image.format not in ("JPEG", "PNG") or image.width * image.height > 4_000_000:
                    raise ValueError("Expected JPEG/PNG up to four million pixels")
                image.load()
                return ImageOps.exif_transpose(image).convert("RGB")
    except (ValueError, binascii.Error, OSError, UnidentifiedImageError,
            Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise HTTPException(422, "Invalid or oversized JPEG/PNG image") from exc


def create_app(model_path=None, device="cuda", dtype="bfloat16", temperature=1.0,
               max_side=384, predictor=None):
    if not math.isfinite(temperature) or temperature <= 0 or not 64 <= max_side <= 1024:
        raise ValueError("Invalid temperature or max_side")
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        if predictor is not None:
            app.state.predict = predictor
        else:
            import torch
            from decider.infer import Example, Q
            from decider.vision import VisionDecisionModel
            from huggingface_hub import snapshot_download

            if device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA GPU is unavailable; check PyTorch and the NVIDIA driver")
            path = model_path or snapshot_download(MODEL_ID, revision=MODEL_REVISION)
            model = VisionDecisionModel(path, dtype=getattr(torch, dtype), grad_ckpt=False).to(device).eval()

            def predict(image, request):
                context = (f"Goal: {request.goal}\n"
                           "The image is from a robot's forward-facing camera. "
                           "Choose a short local path that makes progress toward the goal. "
                           "All moving paths travel forward; curves do not rotate in place. "
                           "Gentle curves have a 1 meter radius; tighter curves have a 0.5 meter radius. "
                           "Move forward when the nearby path ahead is visibly clear and advances the goal. "
                           "Choose a curve toward the goal when its visible swept route is clear. "
                           "If the target is outside the view, consider a clear curved route to look for it; "
                           "its absence alone is not a reason to stop. "
                           "Stop when no immediate movement is clear of nearby obstacles, "
                           "or the image is too unclear to assess the immediate path. "
                           "Select goal reached only when the image shows the goal is already satisfied.")
                example = Example(context, [Q(QUESTION, list(OPTIONS), 0)])
                with torch.inference_mode():
                    inputs = model.prepare([(image, example)])
                    logits = model.slot_logits(inputs)[0, :len(ACTIONS)]
                    probabilities = torch.softmax(logits.float() / temperature, -1).cpu().tolist()
                return dict(zip(ACTIONS, probabilities))

            app.state.predict = predict
        yield

    app = FastAPI(lifespan=lifespan)

    @app.middleware("http")
    async def limit_body(request: Request, call_next):
        from starlette.responses import JSONResponse
        if request.method == "POST":
            body = bytearray()
            async for chunk in request.stream():
                if len(body) + len(chunk) > MAX_BODY:
                    return JSONResponse({"detail": "Request too large"}, status_code=413)
                body.extend(chunk)
            request._body = bytes(body)
        return await call_next(request)

    @app.get("/health")
    def health():
        return {"ready": True, "model": model_path or MODEL_ID,
                "revision": "local" if model_path else MODEL_REVISION,
                "candidates": list(ACTIONS)}

    @app.post("/decide")
    def decide(request: DecisionRequest):
        if not request.goal.strip():
            raise HTTPException(422, "Goal must not be blank")
        # ponytail: one GPU request at a time; add batching only if multiple robots need it.
        if not lock.acquire(blocking=False):
            raise HTTPException(503, "Inference busy; no request queue")
        try:
            started = time.perf_counter()
            image = decode_image(request.image_base64)
            image.thumbnail((max_side, max_side))
            probabilities = validate_probabilities(app.state.predict(image, request))
            result = {"request_id": request.request_id, "probabilities": probabilities,
                      "inference_ms": (time.perf_counter() - started) * 1000}
            return result
        except HTTPException:
            raise
        except Exception as exc:
            logging.exception("Inference failed")
            raise HTTPException(503, "Inference failed; see server log") from exc
        finally:
            lock.release()

    return app


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-path", help="Optional local snapshot instead of the pinned Hub model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Logit temperature; 1.0 follows the upstream vision example, not robot calibration")
    parser.add_argument("--max-side", type=int, default=384)
    args = parser.parse_args()
    uvicorn.run(create_app(args.model_path, args.device, args.dtype, args.temperature, args.max_side),
                host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()

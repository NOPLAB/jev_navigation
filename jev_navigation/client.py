"""Bounded HTTP client, also used by the image benchmark."""

import base64
import json
from urllib.request import Request, urlopen

from .policy import validate_probabilities


def decide(url, image, goal, previous_action, request_id, timeout):
    body = json.dumps({
        "request_id": request_id,
        "image_base64": base64.b64encode(image).decode("ascii"),
        "goal": goal,
        "previous_action": previous_action,
    }).encode("utf-8")
    request = Request(url.rstrip("/") + "/decide", body,
                      {"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=timeout) as response:
        raw = response.read(65537)
    if len(raw) > 65536:
        raise ValueError("Inference response too large")
    result = json.loads(raw)
    if not isinstance(result, dict) or type(result.get("request_id")) is not int:
        raise ValueError("Invalid inference response")
    if result["request_id"] != request_id:
        raise ValueError("Inference request ID mismatch")
    validate_probabilities(result.get("probabilities"))
    return result

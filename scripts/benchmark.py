"""Send one real image repeatedly; reports HTTP latency, not robot success."""

import argparse
import json
import math
import statistics
import time
from pathlib import Path

from jev_navigation.client import decide


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--goal", default="Approach the red chair and stop before touching it.")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    image = args.image.read_bytes()
    decide(args.url, image, args.goal, "stop", 0, args.timeout)  # Warmup excluded.
    timings = []
    for index in range(args.runs):
        start = time.perf_counter()
        result = decide(args.url, image, args.goal, "stop", index + 1, args.timeout)
        elapsed = (time.perf_counter() - start) * 1000
        timings.append(elapsed)
        print(json.dumps({"http_ms": elapsed, **result}))
    print(json.dumps({"runs": args.runs, "p50_http_ms": statistics.median(timings),
                      "p95_http_ms": sorted(timings)[math.ceil(.95 * len(timings)) - 1]}))


if __name__ == "__main__":
    main()

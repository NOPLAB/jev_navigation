# jev_navigation

ROS 2 Humble visual action selection with `Mapika/decider-2b-vision`.
One ROS node sends camera images to one local inference server and publishes
`geometry_msgs/Twist` directly to `/cmd_vel`. There is no command gate node,
Nav2 integration, map, or navigation-specific model training.

```text
/camera/image_raw -> decision_node <-> HTTP inference server
                         |
                         +-> /cmd_vel
```

Author: nop <noplab90@gmail.com>. Code license: BSD-3-Clause.
The upstream decider code and model retain their Apache-2.0 licenses.

## Requirements

- Ubuntu 22.04 / ROS 2 Humble (including WSL 2).
- `uv`, Git, an NVIDIA driver, and a CUDA-compatible PyTorch installation for inference.
- A forward-facing camera publishing timestamped `sensor_msgs/Image` messages.
- A differential-drive base accepting `geometry_msgs/Twist`, with its own command timeout.

The BF16 model weights are approximately 4.1 GB. An RTX 3060 12 GB is a candidate,
not a validated performance target. Image processing and intermediate tensors
require additional memory. Start with one image, batch size one, and 384-pixel
maximum side length. CPU execution is available for debugging, not real-time control.

ROS uses Ubuntu's Python 3.10. The inference server uses a separate Python 3.11+
`.venv`, because the upstream decider package requires Python >=3.11.
`uv.toml` keeps uv cache files under `.venv/uv-cache`.

## Inference server

Run from the repository root, inside Linux/WSL:

```bash
uv --no-cache venv --python 3.11 .venv
export UV_CACHE_DIR="$PWD/.venv/uv-cache"
uv pip install --python .venv/bin/python torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r inference/requirements.txt
.venv/bin/python -m inference.server
```

Use a PyTorch CUDA build compatible with your driver. The command above selects
CUDA 12.8 wheels; it is not a claim that every installed driver supports them.
Run `nvidia-smi` first. The server refuses to silently fall back from CUDA to CPU.

The default endpoint is `http://127.0.0.1:8000`. Model loading finishes before
`GET /health` succeeds. First startup downloads the pinned model snapshot into
the Hugging Face cache. The uv cache and model cache are different.

The upstream code commit is pinned in `inference/requirements.txt`; the model
revision is pinned in `inference/server.py`. To use a downloaded snapshot:

```bash
.venv/bin/python -m inference.server --model-path /path/to/model
```

The service accepts JPEG/PNG images, bounds request/image sizes, and rejects
concurrent inference with HTTP 503 rather than queuing stale camera frames.
It uses the upstream vision `prepare()` / `slot_logits()` path and does not
generate text. Logit temperature defaults to 1.0, matching the upstream vision
example; this is not calibration for robot navigation.

`POST /decide` accepts:

```json
{
  "request_id": 1,
  "image_base64": "<base64 JPEG or PNG bytes, without a data URL prefix>",
  "goal": "Approach the red chair and stop before touching it.",
  "previous_action": "stop"
}
```

The response contains the same request ID, `inference_ms`, and a `probabilities`
object with exactly `forward`, `left`, `right`, `stop`, and `goal_reached`.

## Build the ROS package

Place this repository at `~/ros2_ws/src/jev_navigation`. Use a shell without
the inference virtual environment activated:

```bash
source /opt/ros/humble/setup.bash
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select jev_navigation
source install/setup.bash
ros2 launch jev_navigation navigation.launch.py
```

The server is a separate process; the ROS launch file starts only one ROS node.
Default camera input is `/camera/image_raw`; remap it if needed:

```bash
ros2 launch jev_navigation navigation.launch.py image_topic:=/camera/color/image_raw
```

The node starts disabled and with `dry_run=true`. Enable camera decisions with:

```bash
ros2 service call /jev/enable std_srvs/srv/SetBool '{data: true}'
ros2 topic echo /jev/status
```

Dry run publishes zero velocity while reporting the selected action and probabilities.
To permit movement, restart the launch with `dry_run:=false`, then explicitly enable.
The node always publishes to its configured command topic, including zero commands
while disabled; do not run a competing velocity publisher on that topic.

```bash
ros2 launch jev_navigation navigation.launch.py dry_run:=false
ros2 service call /jev/enable std_srvs/srv/SetBool '{data: true}'
# Stop:
ros2 service call /jev/enable std_srvs/srv/SetBool '{data: false}'
```

Change the goal only while disabled:

```bash
ros2 param set /decision_node goal 'Approach the blue box and stop before touching it.'
```

Other parameters are startup-only. Edit `config/navigation.yaml` or supply a
custom YAML with `config:=/absolute/path/navigation.yaml`. The launch argument
`dry_run` overrides the YAML value. English goals match the model's documented
language. Goal reaching is a model judgment, not an independently verified distance.

## Command behavior

- `forward`: positive linear x; `left` / `right`: positive / negative angular z.
- `stop`: zero velocity. `goal_reached`: zero velocity and disable until re-enabled.
- Low maximum probability or a small top-two margin produces `stop`.
- Only one inference request is in flight. Pending camera images are replaced by newer ones.
- The command lifetime starts at the source image time, not the response time.
  Publishing never renews it. Expired results, invalid timestamps/probabilities,
  HTTP failures, and inference errors cannot start motion.
- Disable/re-enable invalidates outstanding results and requires a fresh camera frame.
- A steady-clock timer publishes at 20 Hz even while inference runs or ROS time pauses.
  A backwards ROS clock jump disables the node. Camera stamps must use the same
  ROS clock as the node; zero stamps and stamps >50 ms into the future are rejected.

There is no `/scan` check or independent obstacle avoidance. Model probabilities
are not collision probabilities. Configure the base driver's command timeout:
a crashed process cannot publish a stop. Begin with wheels lifted or simulation,
then supervised low-speed trials with an independent emergency stop.

## Benchmark and tests

Install the package into the inference environment for the benchmark CLI:

```bash
uv pip install --python .venv/bin/python -e .
.venv/bin/python scripts/benchmark.py photo.jpg --runs 20
```

One warmup is excluded. The script prints decisions, server time, and end-to-end
HTTP p50/p95. Check VRAM separately with `nvidia-smi`. Tune image size and speeds
against measurements; do not increase `command_ttl` merely to hide slow inference.

The small automated suite exercises command expiration, late results after a
restart, invalid/uncertain probabilities, and the HTTP contract with a fake
predictor. It does not download weights or prove model quality:

```bash
uv pip install --python .venv/bin/python fastapi uvicorn pillow httpx
.venv/bin/python -m unittest discover -s test -v
```

Run the ROS/HTTP integration check with Ubuntu's system Python:

```bash
source /opt/ros/humble/setup.bash
RUN_ROS_TESTS=1 python3 -m unittest discover -s test -p test_ros.py -v
```

This uses ROS domain 173 and `/jev_test/*` topics, never the robot's `/cmd_vel`.
It checks direct velocity publication, camera-loss stopping, expired inference,
and disable/re-enable rejection of old results using a mock HTTP server.

Validated on WSL 2 Ubuntu 22.04 with ROS 2 Humble: three control/API tests
(Python 3.11.16), one ROS/HTTP integration test (system Python 3.10.12),
`colcon build --symlink-install`, and installed launch argument loading passed.
Actual model loading, GPU inference, latency, and physical navigation remain untested.
The development environment also has uv at `.venv/bin/uv`; use that path if uv
is not on your WSL PATH. Python dependencies are not fully locked; the upstream
decider code and model revisions are pinned.

## Files

```text
jev_navigation/decision_node.py  ROS image/HTTP/velocity loop
jev_navigation/policy.py         Action contract and command lifetime
jev_navigation/client.py         Shared bounded HTTP client
inference/server.py             Standalone vision inference server
inference/requirements.txt      Upstream code pin and server dependencies
config/navigation.yaml         Initial experiment parameters
launch/navigation.launch.py    One-node ROS launch
scripts/benchmark.py            Real-image latency probe
test/test_control.py            Control and API checks
test/test_ros.py                Isolated real ROS/HTTP check
```

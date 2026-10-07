# ai-vision-ros2-demo (qualcomm branch)

Qualcomm Hexagon NPU (QNN) variants of the `ai-vision-ros2` perception snap
for the Robotics team physical-AI demo. Same snap name, same `perception`
service, swapped via `snap refresh ai-vision-ros2 --channel=...` with **zero
ROS-side restart** — see
[`docs/ai_vision_ros2_qualcomm.md`](docs/ai_vision_ros2_qualcomm.md) for the
full build/install/troubleshooting guide.

| Channel | Package | Behaviour |
| --- | --- | --- |
| `latest/qualcomm/stable` | [`so101_depth_demo/`](so101_depth_demo) | Depth Anything V2 (Small) proximity, ONNX Runtime **QNN HTP EP** |
| `latest/qualcomm/edge` | [`so101_owl_demo/`](so101_owl_demo) | **OWL-ViT** open-vocabulary detection, single fused ONNX graph on QNN HTP EP |

[`so101_yolo_demo/`](so101_yolo_demo) (CPU YOLOv8n) is also present for parity
with the CPU track but is **not** published on a `qualcomm/*` channel — the
Qualcomm `edge` slot is OWL-ViT instead.

Both perception nodes publish `/safety/protective_stop` (`std_msgs/Bool`)
directly; the downstream enforcement nodes (`safety_pause_bridge` /
`trajectory_safety_gate`, in the separate `so101-ros-physical-ai` monorepo's
`so101_safety` package) are perception-agnostic and just subscribe to that
topic.

## Other branches in this repo

| Branch | Hardware | Channels |
| --- | --- | --- |
| [`main`](../../tree/main) | CPU / ONNX Runtime | `latest/stable` (depth), `latest/edge` (yolo) |
| [`nvidia`](../../tree/nvidia) | NVIDIA Jetson Orin, TensorRT | `latest/nvidia/stable` (depth), `latest/nvidia/edge` (NanoOWL) |
| `qualcomm` (this branch) | Qualcomm Hexagon NPU, QNN | `latest/qualcomm/stable` (depth), `latest/qualcomm/edge` (OWL-ViT) |

Non-AI packages (bringup, teleop, safety, idle-demo, MoveIt config) live in
the separate `so101-ros-physical-ai` monorepo this was extracted from.

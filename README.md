# ai-vision-ros2-demo

CPU / ONNX Runtime variant of the `ai-vision-ros2` perception snap for the
Robotics team physical-AI demo — one snap name, two channels, swapped via
`snap refresh ai-vision-ros2 --channel=...` with **zero ROS-side restart**.
See [`docs/ai_vision_ros2_channel_demo.md`](docs/ai_vision_ros2_channel_demo.md)
for the full build/publish/channel-swap walkthrough.

| Channel | Package | Behaviour |
| --- | --- | --- |
| `latest/stable` | [`so101_depth_demo/`](so101_depth_demo) | Depth Anything V2 (Small) monocular depth proximity |
| `latest/edge` | [`so101_yolo_demo/`](so101_yolo_demo) | YOLOv8n person/object detection |

Both variants bundle their ONNX model weights directly in the snap, share the
same app/service name (`perception`), and compute/publish
`/safety/protective_stop` (`std_msgs/Bool`) themselves — the downstream
enforcement nodes (`safety_pause_bridge` / `trajectory_safety_gate`, in the
separate `so101-ros-physical-ai` monorepo's `so101_safety` package) are
perception-agnostic and just subscribe to that topic.

## Other branches in this repo

This repo has one branch per target hardware; all three publish the exact
same `ai-vision-ros2` snap name under different channel tracks:

| Branch | Hardware | Channels |
| --- | --- | --- |
| `main` (this branch) | CPU / ONNX Runtime | `latest/stable` (depth), `latest/edge` (yolo) |
| [`nvidia`](../../tree/nvidia) | NVIDIA Jetson Orin, TensorRT | `latest/nvidia/stable` (depth), `latest/nvidia/edge` (NanoOWL) |
| [`qualcomm`](../../tree/qualcomm) | Qualcomm Hexagon NPU, QNN | `latest/qualcomm/stable` (depth), `latest/qualcomm/edge` (OWL-ViT) |

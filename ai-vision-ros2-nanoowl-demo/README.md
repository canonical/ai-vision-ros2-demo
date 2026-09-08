# ai_vision_ros2_nanoowl_demo — NanoOWL (OWL-ViT) protective-stop vision snap for ROS 2

A Jetson-focused ROS 2 snap that uses NanoOWL (OWL-ViT) open-vocabulary object
detection to monitor a camera feed. Configure a text prompt for the object to
watch for, such as `"a hand"` or `"a face"`; when a matching detection overlaps
the configured safety region, the snap publishes a protective-stop signal.

The snap runs NanoOWL with PyTorch, torch2trt, and TensorRT on the Jetson GPU.
Its image-encoder TensorRT engine is built on-device on first run and can be
bundled for faster deployment.

## What it does

`owl_detect_node` subscribes to a camera image topic, runs NanoOWL (OWL-ViT) for
a configurable **text prompt** (default `"a hand"`), and publishes:

- `/camera/detections/visualization` (annotated `Image`, with the safety ROI)
- `/perception/detections` (`vision_msgs/Detection2DArray`)
- `/safety/protective_stop` (`std_msgs/Bool`) — ROI overlap on the prompt-matched
  detections + hysteresis debounce (the enforcement contract; always published)

The heavy OWL image encoder runs on the Jetson GPU via a **TensorRT engine built
on-device on first run** (cached in `$SNAP_COMMON/trt_engines`, and optionally
bundled — see `prebuilt_engines/`), with a pure-torch fallback. OWL-ViT weights
are pulled from Hugging Face on first run (cached in `$SNAP_COMMON/hf`).

## Build (on the Jetson)

Requires JetPack 6.x (L4T r36.x); the snap pins CUDA 12.6 / cuDNN 9.3 /
TensorRT 10.3 and the `torch 2.8.0` / `torchvision 0.23.0` Jetson wheels.

```bash
cd ai-vision-ros2-nanoowl-demo
SNAPCRAFT_ENABLE_EXPERIMENTAL_EXTENSIONS=1 snapcraft pack --use-lxd --verbose
sudo snap install --dangerous ./ai-vision-ros2_1.0.0-owl-nvidia_arm64.snap
# GPU interfaces:
sudo snap connect ai-vision-ros2:tegra-gpu
sudo snap connect ai-vision-ros2:hardware-observe
sudo snap connect ai-vision-ros2:opengl
sudo snap connect ai-vision-ros2:kernel-module-observe
sudo snap start ai-vision-ros2.perception
```

To bundle the on-device-built engine so a fresh install skips the ~1–2 min
build, copy it into `prebuilt_engines/` before packing:

```bash
cp /var/snap/ai-vision-ros2/common/trt_engines/owl_image_encoder_patch32.engine \
   prebuilt_engines/
```

## Configure

The prompt (what to detect / what stops teleop) is switchable via `snap set` (a file the node watches) or a ROS topic:

```bash
sudo snap set ai-vision-ros2 prompt="a face"
ros2 topic pub -1 /perception/prompt std_msgs/msg/String "{data: 'a hand'}"
```

Other keys: `threshold`, `owl-model`, `input-image-topic`, `use-compressed`
(default `true` — subscribe to the JPEG image to cut DDS/memory load),
`output-image-topic`, `output-detections-topic`, `min-period-s`, `stop-topic`,
`prompt-topic`, `roi`, `min-overlap-ratio`, `frames-to-block`,
`frames-to-clear`, plus `namespace`/`node-name`/`ros-domain-id`/
`rmw-implementation`. Only prompt changes are live; other keys restart the node.

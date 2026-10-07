# ai-vision-ros2 — Qualcomm (Hexagon NPU / QNN) variants

This is the **Qualcomm** counterpart to the CPU and NVIDIA `ai-vision-ros2`
snaps. On the `latest/qualcomm/*` track there are **two** NPU variants that
share the snap name and the `perception` service, so you swap the whole
behaviour with a channel refresh and **zero ROS-side restart**:

| Channel | Built from | Behaviour |
| --- | --- | --- |
| `latest/qualcomm/stable` | `so101_depth_demo/snap/snapcraft.yaml` | Depth Anything V2 (Small) proximity, ONNX Runtime **QNN HTP EP** |
| `latest/qualcomm/edge` | `so101_owl_demo/snap/snapcraft.yaml` | **OWL-ViT open-vocabulary** detection of a configurable text prompt (default `"a hand"`), ONNX Runtime **QNN HTP EP** |

```bash
snap refresh ai-vision-ros2 --channel=qualcomm/stable   # -> depth proximity
snap refresh ai-vision-ros2 --channel=qualcomm/edge      # -> OWL-ViT prompt
```

Both publish `/safety/protective_stop` (`std_msgs/Bool`) themselves (ROI
proximity for depth; ROI overlap on the prompt-matched detections for OWL-ViT),
so the enforcement nodes (`safety_pause_bridge` / `trajectory_safety_gate`)
never change.

- **Base / ROS distro:** both are `core24` + **ROS 2 Jazzy**
  (`ros2-jazzy-ros-base`). Unlike the NVIDIA track, the Qualcomm QNN wheel
  (`onnxruntime-qnn`) ships a **cp312 aarch64** build, so there is no reason to
  drop to core22/Humble.
- **Accelerator:** ONNX Runtime's **QNN execution provider**, HTP backend
  (Hexagon NPU), over fastrpc. The bundled `onnxruntime-qnn` **2.2.0** (QAIRT
  **2.46**) is version-matched to the board's on-device Qualcomm SDK. A
  mismatched QAIRT makes the QNN backend reject the SoC
  (`QNN_DEVICE_ERROR` / "unsupported platform").

---

## OWL-ViT variant (`latest/qualcomm/edge`, `so101_owl_demo`)

OWL-ViT is an open-vocabulary, two-tower (image + text) detector. Here it runs
as a **single fused ONNX graph** (Qualcomm AI-Hub's export) on the Hexagon NPU
via ONNX Runtime's QNN HTP EP — **not** a PyTorch/torch2trt stack like the
Jetson NanoOWL sibling. Inputs are the preprocessed image plus the **CLIP-
tokenised text prompt**; outputs are 576 candidate boxes + sigmoid scores which
the node dequantises, score-thresholds and NMSes (NMS is not in the graph).

Configure the prompt (the whole point — type what should be detected, and
therefore what stops teleop, without rebuilding anything):

```bash
sudo snap set ai-vision-ros2 prompt="a hand"       # single prompt (v1)
sudo snap set ai-vision-ros2 score-threshold=0.1    # detection score threshold
```

A **prompt-only** change does **not** restart the daemon: the configure hook
writes `$SNAP_DATA/prompt.txt`, and the running node polls + re-tokenises it
live (~ms). Any other key change restarts the daemon. You can also push a prompt
over ROS:

```bash
ros2 topic pub --once /perception/prompt std_msgs/String '{data: "a cup"}'
```

Other keys: `input-image-topic`, `use-compressed` (default `true` — subscribe
to `<input-image-topic>/compressed`, `sensor_msgs/CompressedImage`, instead of
raw `Image`), `output-image-topic`,
`output-detections-topic`, `iou-threshold`, `max-detections`, `min-period-s`,
`intra-op-threads`, `qnn-backend-path` (`htp`/`gpu`/`cpu`),
`qnn-context-cache-path`, `execution-providers`, `log-severity-level`,
`stop-topic`, `roi`, `min-overlap-ratio`, `frames-to-block`, `frames-to-clear`,
plus `namespace`/`node-name`/`ros-domain-id`/`rmw-implementation`. The uint16
quant/dequant params (`input-quant-scale`, `box-dequant-scale`,
`score-dequant-scale`, …) default to the bundled model's `metadata.json`
values and rarely need setting.

The OWL-ViT model (`google/owlvit-base-patch32` checkpoint, ViT-B/32 CLIP
backbone) and its CLIP tokenizer are **bundled in the snap** at build time —
no AI-Hub account, no separate model snap, no network at runtime.

> **First start & QNN context caching:** the first-ever start compiles the QNN
> graph (~40 s) and writes an **EPContext** binary (~180 MB) to
> `qnn-context-cache-path` (defaults to `$SNAP_COMMON/qnn_ctx/owl_vit_ctx.onnx`).
> Every subsequent start — restarts, reboots, config changes — reloads that
> cache in **~1 s**, then runs at ~23 ms per inference **fully on the Hexagon
> NPU**. A cache left over from an incompatible QNN/QAIRT version is
> auto-detected (load fails) and regenerated. Set `qnn-context-cache-path=""` to
> disable caching (always recompile).

---

## Depth variant (`latest/qualcomm/stable`, `so101_depth_demo`)

Runs the `depth_anything_node` with ONNX Runtime's QNN HTP EP. Supports two
bundled models via `snap set model=default|aihub`:

- `default` — HuggingFace fp32 Depth Anything V2 Small (308×308), partial-HTP,
  ~17 ms.
- `aihub` — Qualcomm AI-Hub w8a16 model (518×518), full-HTP, ~33 ms.

It uses the **same EPContext context-binary caching** as the OWL-ViT variant
(`qnn-context-cache-path`, default under `$SNAP_COMMON/qnn_ctx/`, one file per
bundled model): the first start compiles + caches (~12 s for `aihub`), later
starts reload in ~0.3 s. Switching `model=default|aihub` keeps a separate cache
per model, so toggling back and forth never recompiles.

See the depth model card / `so101_depth_demo/README.md` for its config keys.

---

## Prerequisites (on the Qualcomm target)

- A Qualcomm board running the **Ubuntu-on-Qualcomm IoT** stack (tested on the
  **IQ-9075 EVK**, SoC QCS9075, HTP arch V73). The snap stages the **fastrpc
  userspace driver** (`qcom-fastrpc1`/`-dev`, providing `libcdsprpc.so`) from
  the **Qualcomm IoT PPA** (`ppa:ubuntu-qcom-iot/qcom-ppa`); the QNN provider +
  backend libs (incl. the Hexagon skels) ship inside the `onnxruntime-qnn`
  wheel.
- `snapd`, and `snapcraft` for building. Build **natively on the board**
  (arm64) under LXD.

### Device access (strict confinement)

The snaps declare a `custom-device` slot (`qcom-accel`) naming the accelerator
nodes and a matching plug. `custom-device` udev-tags the nodes, which is what
the mandatory core24 device cgroup requires:

- `/dev/fastrpc-cdsp` (Hexagon DSP / NPU) — the HTP backend needs this. It is
  owned `fastrpc:fastrpc` with `others=r`, and the confined root daemon opens it
  **O_RDONLY** (which `libcdsprpc` uses), so **no udev/permission change is
  required**.
- `/dev/kgsl-3d0` (Adreno GPU) — kept for the optional `qnn-backend-path=gpu`
  backend.

For a store install the interface auto-connects; for a local `--dangerous`
install connect it manually:

```bash
sudo snap connect ai-vision-ros2:qcom-accel-plug ai-vision-ros2:qcom-accel
```

---

## Build (natively on the board, under LXD)

Each variant is built from its own package directory (the snap project root is
that package dir), using snapcraft's conventional file names
(`snap/snapcraft.yaml`, `snap/hooks/`, `snap/local/`). The
`ros2-jazzy-ros-base` extension needs the experimental-extensions flag:

```bash
# OWL-ViT variant -> qualcomm/edge  (this repo)
cd so101_owl_demo
SNAPCRAFT_ENABLE_EXPERIMENTAL_EXTENSIONS=1 snapcraft pack --verbosity=trace

# Depth variant -> qualcomm/stable
cd ../so101_depth_demo
SNAPCRAFT_ENABLE_EXPERIMENTAL_EXTENSIONS=1 snapcraft pack --verbosity=trace
```

Install + smoke test:

```bash
bash so101_owl_demo/snap/smoke_tests.sh
```

---

## The QNN recipe (why these exact pins)

The same hard-won recipe as the depth variant applies (see the depth notes and
`AGENTS.md`):

1. Pin `onnxruntime==1.24.4` + `onnxruntime-qnn==2.2.0` (QAIRT **2.46**, matches
   the board SDK). The default newer wheel (QAIRT 2.49) gives
   `PLATFORM_NOT_SUPPORTED`.
2. Use the ORT **plugin-EP API** — `register_execution_provider_library` +
   `get_ep_devices` + `add_provider_for_devices(devs, {"backend_type":"htp"})`.
   Passing `providers=[...]` to `InferenceSession` is silently ignored for
   plugin EPs.
3. **Static input shapes** — QNN can't finalise a dynamic graph. The AI-Hub
   OWL-ViT export is already static (`1x3x768x768` / `1x16`).
4. Stage **`qcom-fastrpc1`** (`libcdsprpc.so`, dlopen'd by `libQnnHtp.so`).
5. Set `ADSP_LIBRARY_PATH` to the `onnxruntime_qnn` wheel dir (ships the V73
   skel) — done in the launcher.
6. `custom-device` slot for `/dev/fastrpc-cdsp` (opens O_RDONLY confined — no
   udev/chmod needed).

OWL-ViT's AI-Hub export is validated with QAIRT 2.45; it loads and runs **fully
on the HTP** under the board's 2.46 stack (verified: ~23 ms/inference, CPU-EP
fallback disabled passes).

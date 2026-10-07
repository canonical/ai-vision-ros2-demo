# ai-vision-ros2 — NVIDIA (Jetson / TensorRT) variants

This is the **NVIDIA Jetson** counterpart to the CPU `ai-vision-ros2` snap. On
the `latest/nvidia/*` track there are **two** GPU variants that share the snap
name and the `perception` service, so you swap the whole behaviour with a
channel refresh and **zero ROS-side restart**:

| Channel | Built from | Behaviour |
| --- | --- | --- |
| `latest/nvidia/stable` | `so101_depth_demo/snap/snapcraft.yaml` | Depth Anything V2 (Small) proximity, ONNX Runtime **TensorRT EP** |
| `latest/nvidia/edge` | [`ai-vision-ros2-nanoowl-demo/snap/snapcraft.yaml`](../ai-vision-ros2-nanoowl-demo/snap/snapcraft.yaml) | **NanoOWL (OWL-ViT) open-vocabulary** detection of a configurable text prompt (default `"a hand"`), image encoder on TensorRT via `torch2trt` |

```bash
snap refresh ai-vision-ros2 --channel=nvidia/stable   # -> depth proximity
snap refresh ai-vision-ros2 --channel=nvidia/edge      # -> NanoOWL prompt
```

Both publish `/safety/protective_stop` (`std_msgs/Bool`) themselves (ROI
proximity for depth; ROI overlap on the prompt-matched detections for NanoOWL),
so the enforcement nodes never change.

- **Snap name / service / config keys:** the depth variant keeps the CPU snap's
  keys; the NanoOWL variant swaps the depth-specific keys for `prompt` /
  `threshold` (see below).
- **Base / ROS distro:** both are `core22` + **ROS 2 Humble**
  (`ros2-humble-ros-base` extension). See "Why core22 + Humble" below.

---

## NanoOWL variant (`latest/nvidia/edge`, `ai-vision-ros2-nanoowl-demo`)

NanoOWL is **not** an ONNX Runtime model — OWL-ViT is an open-vocabulary,
two-tower (image + text) detector, so this variant runs a PyTorch stack
(`transformers` + `torch2trt` + `nanoowl`) with the heavy image encoder
accelerated by a **TensorRT engine built on-device on first run**. The OWL-ViT
weights are downloaded from Hugging Face on first run and cached in
`$SNAP_COMMON/hf`; the engine is cached in `$SNAP_COMMON/trt_engines`. If the
engine can't be built/loaded the node falls back to a slower pure-torch image
encoder (the daemon still works).

Configure the prompt (this is the whole point — type what should be detected,
and therefore what stops teleop, without rebuilding anything):

```bash
sudo snap set ai-vision-ros2 prompt="a hand"     # comma-separated for multiple
sudo snap set ai-vision-ros2 threshold=0.1        # detection score threshold
```

Other keys: `input-image-topic`, `output-image-topic`,
`output-detections-topic`, `owl-model`, `min-period-s`, `stop-topic`, `roi`,
`min-overlap-ratio`, `frames-to-block`, `frames-to-clear`, plus
`namespace`/`node-name`/`ros-domain-id`/`rmw-implementation`. The image-encoder
engine path and the model path are managed by the snap (not user-settable).

> First run: expect a few minutes while the TensorRT image-encoder engine is
> built and the OWL-ViT weights are downloaded (needs network). Subsequent
> starts reuse both caches and start fast (offline).

---

## Depth variant (`latest/nvidia/stable`, `so101_depth_demo`)

Runs the exact same `depth_anything_node` as the CPU snap, but with ONNX
Runtime's **TensorRT execution provider** so Depth Anything V2 (Small) executes
on the Orin's GPU instead of the CPU. Config keys identical to the CPU snap.

---

## Why core22 + Humble (not core24 + Jazzy like the CPU snap)

Two hard constraints force this:

1. **No prebuilt aarch64 `onnxruntime-gpu` wheel for Python 3.12.** The only
   Jetson ORT-gpu wheels (jetson-ai-lab index) are **cp310 = Python 3.10**.
   `core24` = Ubuntu 24.04 = Python 3.12, which cannot load a cp310 wheel.
   `core22` = Ubuntu 22.04 = Python 3.10, which matches the wheel exactly.
2. **The `ros2-jazzy` extension is core24-only.** On core22 the only ROS 2
   content-sharing extension is `ros2-humble-ros-base`.

This mirrors both reference snaps in this repo (`ai-vision-ros2-nanoowl-demo`,
and the upstream `snap-twin`), which are also core22/Py3.10 for the same reason.

**Interop note (Humble ↔ Jazzy):** the depth node is pure `rclpy` +
`sensor_msgs`/`std_msgs`, so it runs identically on Humble. It exchanges DDS
topics with the rest of the (Jazzy) stack provided the **RMW implementation
and `ROS_DOMAIN_ID` match**. Any publisher feeding it on the Jetson (e.g. the
`usb-cam` snap) must also be a **Humble** build on the same domain.

---

## Prerequisites (on the Jetson)

- NVIDIA Jetson (Orin) with **JetPack 6.x** (L4T r36.x). The snap stages the
  **CUDA/cuDNN/TensorRT runtime** (CUDA 12.6 / cuDNN 9.3 / TensorRT 10.3) from
  NVIDIA's Jetson apt repos, and — separately — the **Tegra GPU driver
  userspace shims** (`libcuda`, `libnvcudla`, `libnvdla_*`, `libnvrm_*`,
  `libnvcucompat`, `libnvsciipc`, …) **version-matched to this device's kernel
  `nvgpu` driver**. On this Canonical Ubuntu-for-Tegra device that means the
  `ubuntu-tegra/updates` PPA (`36.5`); NVIDIA's own repo ships the same libs but
  at `36.4.7`, which does **not** match the kernel — see "GPU driver shims".
- `snapd`, `snapcraft` (for building), and the `ros-humble-ros-base` content
  snap (auto-installed via the extension's `default-provider` when installing
  from the store; for `--dangerous` sideloads install it manually).

> **Driver-version pin:** the shims must match the running nvgpu **kernel**
> driver. `snap/snapcraft.yaml`'s `gpu-driver` part pins them to this device's
> version (`36.5.20260115-0ubuntu1`). If the Jetson's driver updates, bump
> `TEGRA_VER` to match `dpkg-query -W nvidia-tegra-drivers-36-igpu-cuda` and
> rebuild.

---

## Build (on the Jetson)

Each variant is built from its own package directory (the snap project root is
that package dir), using snapcraft's conventional file names
(`snap/snapcraft.yaml`, `snap/hooks/`, `snap/local/`):

```bash
# Depth variant -> nvidia/stable  (this repo)
cd so101_depth_demo   # project root for the snap is this package dir

# NanoOWL variant -> nvidia/edge  (this repo)
cd ai-vision-ros2-nanoowl-demo   # project root for the snap is this package dir

# Build (arm64 native on the Jetson). The ros2-humble-ros-base extension is
# experimental, so enable experimental extensions.
SNAPCRAFT_ENABLE_EXPERIMENTAL_EXTENSIONS=1 snapcraft --use-lxd --verbose
```

> The depth build downloads the Jetson `onnxruntime-gpu` wheel (pinned in
> `snap/snapcraft.yaml`), the Depth Anything ONNX weights from Hugging Face,
> and the pinned Tegra driver-shim debs. The NanoOWL build downloads the pinned
> Jetson `torch 2.8.0` / `torchvision 0.23.0` wheels (jetson-ai-lab cu126 index)
> and `torch2trt`/`nanoowl`/CLIP from git. Both need network access.
>
> The NanoOWL pins target **JetPack 6.2 / L4T r36.5 / CUDA 12.6 / TensorRT
> 10.3** (Python 3.10). `trtexec` comes from `libnvinfer-bin` and the `tensorrt`
> Python module from `python3-libnvinfer` (both 10.3.0.30). If the device's
> JetPack changes, re-pin the torch/torchvision wheels from the matching
> `pypi.jetson-ai-lab.io/jp6/cuXXX` index.

### GPU driver shims (why the `gpu-driver` part exists)

The `gpu-driver` part `apt-get download`s three pinned debs
(`nvidia-tegra-drivers-36-{igpu-cuda,graphic,nvsci}`) from the `ubuntu-tegra`
PPA, extracts them, and copies **only a ~16-lib shim allowlist** into
`$SNAP/usr/lib/aarch64-linux-gnu/nvidia` (put **first** on `LD_LIBRARY_PATH`).
`opengl` provides the GPU device nodes but does **not** inject these driver libs
on classic Ubuntu, so they must be staged.

**Match the shim version to the kernel `nvgpu` driver.** This is the #1 gotcha.
The shims are tightly coupled to the running kernel driver; a skew does **not**
fail at load time — the libs load, then `libnvrm_gpu` reads the chip id as zeros
and dies (`No matching chip spec found for chip Id=0` / `NvRmGpuLibOpen failed`),
and ORT silently falls back to CPU. On this **Canonical Ubuntu-for-Tegra** device
the matched libs live only in the `ubuntu-tegra` PPA (`36.5`), which is why the
part uses the PPA and not NVIDIA's repo (whose `nvidia-l4t-* 36.4.7` mismatches
the kernel). On a stock **NVIDIA JetPack** device it would be the reverse
(NVIDIA's repo matches — which is why Canonical's `tegra-snap-samples` need no
PPA).

**Do not stage the full `nvidia` lib dir.** It contains GPU compute-core libs
(`libnvidia-*`, `ptxjitcompiler`, `nvvm`, `gpucomp`) that **shadow the host
copies snapd injects via the `opengl` interface**, which silently corrupts the
depth output (blank/garbage) even though CUDA initialises. Only the thin driver
shims belong in the snap; the compute core must come from the host. (This is
not a precision/fp16 bug — it was ruled out.)

---

## Install + connect interfaces

GPU access needs two things, and only one of them is your responsibility:

1. **Driver libraries** — you stage them (the version-matched shims above) plus
   the host compute core via the `opengl` interface.
2. **GPU device nodes** — provided by the **`opengl`** interface alone (it
   carries the Tegra iGPU `/dev/nvhost-*`, `/dev/nvmap`,
   `/dev/nvgpu/igpu*/{power,ctrl,prof}` rules + udev cgroup tags). **No
   `system-files` plug is needed** — this was verified on-device: with
   version-matched libs, `opengl` alone runs the demo on the GPU.

`opengl` **auto-connects** on classic Ubuntu, so a store install needs no manual
GPU wiring. For a `--dangerous` sideload:

```bash
sudo snap install --dangerous ./ai-vision-ros2_*.snap   # or: sudo snap install ai-vision-ros2 --channel=latest/nvidia/stable

# ROS 2 content runtime (only needed for --dangerous sideloads)
sudo snap install ros-humble-ros-base
sudo snap connect ai-vision-ros2:ros-humble-ros-base ros-humble-ros-base:ros-humble-ros-base 2>/dev/null || true

# GPU access — opengl usually auto-connects; hardware-observe / kernel-module-observe do not
sudo snap connect ai-vision-ros2:opengl                 # device nodes + host compute-core libs
sudo snap connect ai-vision-ros2:hardware-observe
sudo snap connect ai-vision-ros2:kernel-module-observe

sudo snap start ai-vision-ros2.perception
```

Check what auto-connected with `snap connections ai-vision-ros2`.

> **No `tegra-gpu` / `system-files` plug and no `tegra-libs` plug.** The earlier
> design listed every `/dev/nvgpu/*` node in a `system-files` plug; that turned
> out to be unnecessary — the failure it was working around
> (`chip Id=0` / `NvRmGpuLibOpen`) was a driver-lib **version skew**, not missing
> device access. `opengl` covers the nodes. (If a future workload ever does hit
> `CUDA failure 999` *with* version-matched libs, that's the only case where a
> `system-files` fallback for the extra `/dev/nvgpu/igpu0/*` nodes would be
> warranted.)

---

## Configure

Same keys as the CPU snap, plus two NVIDIA-specific ones:

```bash
sudo snap set ai-vision-ros2 input-image-topic=/my/camera/image_raw
sudo snap set ai-vision-ros2 output-image-topic=/camera/depth/visualization

# NVIDIA-specific (defaults shown):
sudo snap set ai-vision-ros2 execution-providers="TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider"
sudo snap set ai-vision-ros2 engine-cache-dir="/var/snap/ai-vision-ros2/common/trt_engines"
```

`execution-providers` is a comma-separated ORT provider list in priority
order; the configure hook converts it to the `execution_providers` ROS
parameter (a YAML string list).

The engine cache defaults to **`$SNAP_COMMON`** (`/var/snap/ai-vision-ros2/common/trt_engines`),
which is shared across revisions/channels — so the built engine survives
`snap refresh` / channel swaps and is only built once.

## First run: TensorRT engine build

On the **first** run (or after changing the model / providers / input size),
ONNX Runtime's TensorRT provider **builds and caches the engine** under
`engine-cache-dir`. Expect **~1–2 minutes** of one-time startup before frames
flow. Subsequent daemon restarts reuse the cached engine and start fast.

**Prebuilt-engine seeding:** the snap ships a reference engine (built once on an
Orin) under `$SNAP/prebuilt_engines/`. On startup the launcher copies it into
the cache dir with **no-clobber** (`cp -an`), so a matching device skips the
build entirely on first run. If the bundled engine doesn't match this device
(different GPU / TensorRT version / model / precision), ORT validates it,
ignores it, and rebuilds automatically — it's a fast-path optimisation, never a
correctness requirement. A locally rebuilt engine always wins over the bundled
one. To refresh the bundled engine after a model/precision/TRT change, re-copy
`/var/snap/ai-vision-ros2/common/trt_engines/*` into `so101_depth_demo/prebuilt_engines/`
and rebuild (the build itself can't generate an engine — that needs the GPU).

```bash
sudo snap logs ai-vision-ros2 -f        # watch for "providers(active)=[TensorrtExecutionProvider, ...]"
ros2 topic hz /camera/depth/visualization
```

If the log shows only `CPUExecutionProvider` active, TensorRT wasn't usable.
Check, in order: (1) a `cannot open shared object file` cascade (`libnvos.so`,
`libnvdla_compiler.so`, …) → driver shims not staged / not on `LD_LIBRARY_PATH`;
(2) `No matching chip spec found for chip Id=0` / `NvRmGpuLibOpen failed` → the
staged shim **version doesn't match the kernel `nvgpu` driver** (bump `TEGRA_VER`,
see "GPU driver shims"); (3) `opengl` not connected.

---

## Verify the protective stop

As with the CPU variant, the node publishes `/safety/protective_stop`
(`std_msgs/Bool`) computed directly from the depth map (ROI proximity +
hysteresis). The downstream enforcement nodes (`safety_pause_bridge` /
`trajectory_safety_gate` in `so101_safety`) are perception-agnostic and just
subscribe to that topic.

## Troubleshooting

- **`Publisher count: 0` on the input topic** → topic/namespace mismatch with
  the camera snap; check `ros2 topic info <input-image-topic> --verbose`.
- **RTPS / SHM errors** → the launcher already forces UDP-only via
  `FASTRTPS_DEFAULT_PROFILES_FILE`; ensure the camera side does the same.
- **`No matching chip spec found for chip Id=0` / `NvRmGpuLibOpen failed`, then
  CPU fallback** → staged driver-shim **version skew** vs the kernel `nvgpu`
  driver. Match `TEGRA_VER` to `dpkg-query -W nvidia-tegra-drivers-36-igpu-cuda`.
  This looks like a device-access failure but isn't.
- **`libnvos.so` / `libnvdla_compiler.so` / `libcuda.so.1: cannot open shared
  object file`** → driver shims not staged (the `gpu-driver` part) or the dir
  isn't first on `LD_LIBRARY_PATH`. `opengl` does not inject these.
- **`CUDA failure 999` / EPERM at `cudaSetDevice`** (only *with* version-matched
  libs) → `opengl`'s `/dev/nvgpu/igpu*/` coverage (`power,ctrl,prof`) fell short;
  add the missing node(s) via a `system-files` fallback. Rare — rule out a
  version skew first.
- **Depth output blank/garbage only inside the snap** → the full `nvidia` lib
  dir was staged and its compute-core libs shadow the host `opengl` libs; prune
  to the shim allowlist (see "GPU driver shims"). Not a precision/fp16 bug.
- **ImportError about numpy source dir** → actually a missing BLAS/LAPACK; the
  launcher exports `LD_LIBRARY_PATH` for `lapack`/`blas` (same fix as CPU).
- **Humble/Jazzy topics not visible to each other** → different `RMW_IMPLEMENTATION`
  or `ROS_DOMAIN_ID`; align both (`snap set ai-vision-ros2 ros-domain-id=...`).

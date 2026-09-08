"""
NanoOWL (OWL-ViT) open-vocabulary detection node — NVIDIA Jetson / TensorRT
============================================================================
Subscribes to a ROS image topic, runs NanoOWL (OWL-ViT) open-vocabulary
detection for a configurable TEXT PROMPT (default "a hand"), and republishes
an annotated visualisation image (for humans, with the safety ROI overlaid),
machine-readable detections, and a Bool protective-stop signal computed
directly from the (prompt-matched) detections.

The protective-stop signal is determined by overlap between prompt-matched
detections and a configured region of interest (ROI). Hysteresis debounce
prevents the signal from flickering due to isolated noisy frames. Consumers can
subscribe to the configurable ``stop_topic`` to act when the node requests a
protective stop.

NanoOWL is a PyTorch stack: OWL-ViT (``transformers``) whose heavy image encoder
is accelerated with a TensorRT engine (``torch2trt``). 
The image-encoder engine is device and TensorRT-version-specific,
so it is BUILT ON-DEVICE on first run if it is missing. If the engine
cannot be built/loaded, the node falls back to running the image encoder in
plain torch.

Subscribes:
  <input_image_topic>        sensor_msgs/Image           (rgb8 | bgr8 | mono8 | rgba8 | bgra8)

Publishes:
  <output_image_topic>       sensor_msgs/Image           encoding=rgb8   (boxes + labels + ROI/trigger overlay)
  <output_detections_topic>  vision_msgs/Detection2DArray                (for machine consumers)
  <stop_topic>                std_msgs/Bool                              True = protective stop requested

Parameters:
  input_image_topic       (string) camera image topic to subscribe to
  output_image_topic      (string) annotated visualisation topic to publish
  output_detections_topic (string) Detection2DArray topic to publish
  prompt                  (string) open-vocabulary text prompt(s); comma-separated
                                   for multiple (default "a hand")
  threshold               (float)  minimum detection score to keep (default 0.1)
  owl_model               (string) OWL-ViT HuggingFace model id
                                   (default "google/owlvit-base-patch32")
  image_encoder_engine    (string) path to the TensorRT image-encoder engine
                                   ("" = pure torch, no TensorRT). Built here on
                                   first run if the path is set but missing.
  min_period_s            (float)  minimum seconds between inferences (throttle)
  stop_topic              (string) Bool protective-stop output (default /safety/protective_stop)
  roi                     (string) "x1,y1,x2,y2" normalised center ROI
  min_overlap_ratio       (float)  minimum fraction of a detection's box that
                                   must fall inside the ROI to count
  frames_to_block         (int)    consecutive triggered frames to assert stop
  frames_to_clear         (int)    consecutive clear frames to release stop
"""

import array
import os
import time
from collections import deque

import cv2
import numpy as np
import PIL.Image
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, String
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
    Pose2D,
)

from nanoowl.owl_predictor import OwlPredictor

# Deterministic per-label BGR colour for box drawing (fixed palette, no randomness).
_PALETTE = [
    (56, 56, 255), (49, 210, 207), (10, 249, 72), (255, 194, 0),
    (255, 56, 132), (147, 69, 52), (199, 55, 255), (0, 194, 255),
]


def _parse_roi(value):
    """Parse a normalised "x1,y1,x2,y2" ROI string (or 4-item sequence)."""
    vals = (
        [float(v) for v in value.split(",")]
        if isinstance(value, str)
        else [float(v) for v in value]
    )
    if len(vals) != 4 or not all(0.0 <= v <= 1.0 for v in vals):
        raise ValueError("roi must be 4 normalised values x1,y1,x2,y2 in [0,1]")
    return vals


def _roi_pixel_bounds(roi, width, height):
    x1, y1, x2, y2 = roi
    return int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)


class _HysteresisDebouncer:
    """Debounces a per-frame boolean with separate assert/release windows,
    to avoid flickering the safety-stop state on single noisy frames."""

    def __init__(self, frames_to_block: int, frames_to_clear: int):
        self._n_block = int(frames_to_block)
        self._n_clear = int(frames_to_clear)
        self._hist = deque(maxlen=max(self._n_block, self._n_clear))
        self._state = False

    def update(self, is_triggered: bool) -> bool:
        self._hist.append(bool(is_triggered))
        block = list(self._hist)[-self._n_block:]
        clear = list(self._hist)[-self._n_clear:]
        if len(block) == self._n_block and all(block):
            self._state = True
        elif len(clear) == self._n_clear and not any(clear):
            self._state = False
        return self._state


class OwlDetectNode(Node):

    def __init__(self):
        super().__init__("owl_detect_node")

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter("input_image_topic", "/static_camera/image_raw")
        self.declare_parameter("use_compressed", True)
        self.declare_parameter("output_image_topic", "/camera/detections/visualization")
        self.declare_parameter("output_detections_topic", "/perception/detections")
        self.declare_parameter("prompt", "a hand")
        self.declare_parameter("threshold", 0.1)
        self.declare_parameter("owl_model", "google/owlvit-base-patch32")
        self.declare_parameter("image_encoder_engine", "")
        self.declare_parameter("min_period_s", 0.0)
        self.declare_parameter("stop_topic", "/safety/protective_stop")
        self.declare_parameter("prompt_topic", "/perception/prompt")
        self.declare_parameter("prompt_file", "")
        self.declare_parameter("roi", "0.25,0.2,0.75,0.85")
        self.declare_parameter("min_overlap_ratio", 0.2)
        self.declare_parameter("frames_to_block", 2)
        self.declare_parameter("frames_to_clear", 3)

        in_topic = self.get_parameter("input_image_topic").value
        self._use_compressed = bool(self.get_parameter("use_compressed").value)
        out_topic = self.get_parameter("output_image_topic").value
        det_out_topic = self.get_parameter("output_detections_topic").value
        prompt = str(self.get_parameter("prompt").value)
        self._threshold = float(self.get_parameter("threshold").value)
        owl_model = str(self.get_parameter("owl_model").value)
        engine_path = str(self.get_parameter("image_encoder_engine").value)
        self._min_period = float(self.get_parameter("min_period_s").value)
        stop_topic = self.get_parameter("stop_topic").value
        prompt_topic = self.get_parameter("prompt_topic").value
        self._prompt_file = str(self.get_parameter("prompt_file").value)
        self._roi = _parse_roi(self.get_parameter("roi").value)
        self._min_overlap = float(self.get_parameter("min_overlap_ratio").value)
        n_block = int(self.get_parameter("frames_to_block").value)
        n_clear = int(self.get_parameter("frames_to_clear").value)
        self._debouncer = _HysteresisDebouncer(n_block, n_clear)

        # If a prompt file is configured and already populated (the snap's
        # configure hook writes it on `snap set prompt=...`), prefer its content
        # as the initial prompt so a `snap set` before startup is honoured.
        self._prompt_file_mtime = None
        if self._prompt_file and os.path.isfile(self._prompt_file):
            try:
                with open(self._prompt_file, encoding="utf-8") as fh:
                    file_prompt = fh.read().strip()
                if file_prompt:
                    prompt = file_prompt
                self._prompt_file_mtime = os.path.getmtime(self._prompt_file)
            except OSError:
                pass

        # Comma-separated prompt(s) -> OWL-ViT text query list. The synthetic
        # "class id" of each detection is the index into this list.
        self._labels = [p.strip() for p in prompt.split(",") if p.strip()]
        if not self._labels:
            self._labels = ["a hand"]

        # ── NanoOWL predictor (image encoder on the Jetson GPU) ──────────────
        # Loads a ready-made TensorRT engine if one is present (either bundled
        # in the snap and seeded into the cache, or built on a previous run),
        # so a warm install gets straight to TensorRT inference. Builds it once
        # if missing, and falls back to a slower pure-torch encoder if TensorRT
        # is unavailable.
        self._predictor = self._build_predictor(owl_model, engine_path)
        # Encode the text prompt(s) once up front.
        self._text_encodings = self._predictor.encode_text(self._labels)

        # ── Pub / Sub ───────────────────────────────────────────────────────
        self._pub = self.create_publisher(Image, out_topic, 5)
        self._det_pub = self.create_publisher(Detection2DArray, det_out_topic, 5)
        self._stop_pub = self.create_publisher(Bool, stop_topic, 10)
        # Subscribe with a KEEP_LAST depth-1, BEST_EFFORT ("sensor data") QoS so
        # we always run inference on the FRESHEST camera frame and let DDS drop
        # the backlog. The camera (30 fps) far outpaces OWL-ViT inference
        # (~6 fps); with a deeper queue the node would process stale frames and
        # the displayed detections would visibly lag reality. Depth-1 keeps
        # latency at ~one inference period instead of queue_depth periods.
        image_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        if self._use_compressed:
            comp_topic = in_topic.rstrip("/") + "/compressed"
            self._sub = self.create_subscription(
                CompressedImage, comp_topic, self._compressed_cb, image_qos
            )
            self._sub_topic = comp_topic
        else:
            self._sub = self.create_subscription(
                Image, in_topic, self._image_cb, image_qos
            )
            self._sub_topic = in_topic

        # Live prompt updates: publish a std_msgs/String here to change what is
        # detected (and therefore what stops teleop) WITHOUT restarting the node
        # (a full node restart reloads torch + the TRT engine, ~15s; this path
        # just re-encodes the text prompt, ~milliseconds).
        self._prompt_sub = self.create_subscription(
            String, prompt_topic, self._prompt_cb, 1
        )

        # Second live path: watch a prompt file. The snap's configure hook writes
        # the `snap set prompt=...` value here, so `snap set` also switches the
        # prompt live (no service restart). Polled on a light timer.
        if self._prompt_file:
            self._prompt_timer = self.create_timer(0.5, self._check_prompt_file)

        self._frame_count = 0
        self._last_infer_t = 0.0
        self.get_logger().info(
            f"OwlDetectNode ready — subscribing '{self._sub_topic}'"
            + (" (compressed)" if self._use_compressed else "")
            + ", "
            f"prompt={self._labels} threshold={self._threshold}, "
            f"live prompt updates on '{prompt_topic}' (std_msgs/String)"
            + (f" or file '{self._prompt_file}'" if self._prompt_file else "")
            + ", "
            f"publishing viz '{out_topic}', detections '{det_out_topic}', "
            f"protective stop '{stop_topic}' (overlap>{self._min_overlap})"
        )

    def _apply_prompt(self, text: str) -> bool:
        """Re-encode the OWL-ViT text prompt in place (no model/engine reload).
        Returns True if the prompt actually changed and was applied."""
        labels = [p.strip() for p in (text or "").split(",") if p.strip()]
        if not labels or labels == self._labels:
            return False
        try:
            text_encodings = self._predictor.encode_text(labels)
        except Exception as exc:  # noqa: BLE001 - keep serving on a bad prompt
            self.get_logger().warn(f"Ignoring prompt '{text}': {exc}")
            return False
        # Single-threaded executor => this swap can't race the image callback.
        self._labels = labels
        self._text_encodings = text_encodings
        self.get_logger().info(f"Prompt updated live -> {self._labels}")
        return True

    def _prompt_cb(self, msg: String):
        self._apply_prompt(msg.data)

    def _check_prompt_file(self):
        try:
            mtime = os.path.getmtime(self._prompt_file)
        except OSError:
            return
        if mtime == self._prompt_file_mtime:
            return
        self._prompt_file_mtime = mtime
        try:
            with open(self._prompt_file, encoding="utf-8") as fh:
                text = fh.read().strip()
        except OSError:
            return
        self._apply_prompt(text)

    def _new_predictor(self, owl_model: str, engine_path):
        return OwlPredictor(
            owl_model, device="cuda", image_encoder_engine=engine_path
        )

    def _build_predictor(self, owl_model: str, engine_path: str):
        """Construct the OwlPredictor, preferring a ready-made TensorRT engine.

        Order of preference:
          1. An engine already present at ``engine_path`` (bundled+seeded by the
             snap, or built on a previous run) -> load directly (fast start,
             instant inference, like the depth snap's prebuilt/cached engine).
          2. If that engine is missing OR fails to load (e.g. it was built for a
             different GPU / TensorRT version), (re)build it once on-device.
          3. If TensorRT is unavailable entirely, fall back to the pure-torch
             image encoder so the daemon still serves (just slower).
        """
        # 1. Try an existing engine (bundled/seeded or previously built).
        if engine_path and os.path.exists(engine_path):
            try:
                predictor = self._new_predictor(owl_model, engine_path)
                self.get_logger().info(
                    f"Loaded OWL image-encoder TensorRT engine: {engine_path}"
                )
                return predictor
            except Exception as exc:  # noqa: BLE001 - resilience over strictness
                self.get_logger().warn(
                    f"Existing engine failed to load ({exc}); rebuilding it "
                    f"(likely built for a different GPU / TensorRT version)."
                )
                try:
                    os.remove(engine_path)
                except OSError:
                    pass

        # 2. Build the engine once if we have a target path.
        if engine_path:
            try:
                os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)
                self.get_logger().info(
                    f"Building OWL image-encoder TensorRT engine at {engine_path} "
                    f"(first run, this can take a few minutes)..."
                )
                builder = OwlPredictor(owl_model, device="cuda")
                builder.build_image_encoder_engine(engine_path)
                del builder
                predictor = self._new_predictor(owl_model, engine_path)
                self.get_logger().info("OWL image-encoder engine built and loaded.")
                return predictor
            except Exception as exc:  # noqa: BLE001 - resilience over strictness
                self.get_logger().warn(
                    f"Engine build/load failed ({exc}); falling back to the "
                    f"(slower) torch image encoder."
                )

        # 3. Pure-torch fallback (no TensorRT).
        self.get_logger().info(
            "Using pure-torch image encoder (no TensorRT acceleration)."
        )
        return self._new_predictor(owl_model, None)


    @staticmethod
    def _image_to_bgr(msg: Image) -> np.ndarray:
        """Convert a sensor_msgs/Image into an HxWx3 BGR uint8 array."""
        enc = msg.encoding.lower()
        buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)

        if enc in ("rgb8", "bgr8"):
            img = buf.reshape(msg.height, msg.step // 3, 3)[:, : msg.width, :]
            if enc == "rgb8":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return np.ascontiguousarray(img)
        if enc in ("mono8", "8uc1"):
            gray = buf.reshape(msg.height, msg.step)[:, : msg.width]
            return cv2.cvtColor(np.ascontiguousarray(gray), cv2.COLOR_GRAY2BGR)
        if enc in ("rgba8", "bgra8"):
            img = buf.reshape(msg.height, msg.step // 4, 4)[:, : msg.width, :3]
            if enc == "rgba8":
                img = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2BGR)
            else:
                img = np.ascontiguousarray(img)
            return img
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")

    def _predict(self, bgr_frame: np.ndarray):
        """Run NanoOWL on a BGR frame; return a list of
        (x1, y1, x2, y2, conf, label_id) in original-frame pixel coordinates.
        OwlPredictor.predict returns boxes already mapped back to the input
        image size, so no rescaling is needed."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        pil = PIL.Image.fromarray(rgb)
        output = self._predictor.predict(
            image=pil,
            text=self._labels,
            text_encodings=self._text_encodings,
            threshold=self._threshold,
            pad_square=True,
        )
        boxes = output.boxes.detach().cpu().numpy().reshape(-1, 4)
        scores = output.scores.detach().cpu().numpy().reshape(-1)
        labels = output.labels.detach().cpu().numpy().reshape(-1)

        detections = []
        for (x1, y1, x2, y2), conf, lbl in zip(boxes, scores, labels):
            detections.append(
                (float(x1), float(y1), float(x2), float(y2), float(conf), int(lbl))
            )
        return detections

    def _label_text(self, label_id: int) -> str:
        return self._labels[label_id] if 0 <= label_id < len(self._labels) else str(label_id)


    def _evaluate_stop(self, detections, frame_w: int, frame_h: int):
        """Given this frame's (prompt-matched) detections, return
        (stop_state, best_overlap_ratio). Uses the ACTUAL per-frame camera
        dimensions."""
        ix1, iy1, ix2, iy2 = _roi_pixel_bounds(self._roi, frame_w, frame_h)
        best_overlap = 0.0
        triggered = False
        for x1, y1, x2, y2, _conf, _lbl in detections:
            ox1, oy1 = max(x1, ix1), max(y1, iy1)
            ox2, oy2 = min(x2, ix2), min(y2, iy2)
            inter_w, inter_h = max(ox2 - ox1, 0.0), max(oy2 - oy1, 0.0)
            inter_area = inter_w * inter_h
            box_area = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
            if box_area <= 0.0:
                continue
            overlap_ratio = inter_area / box_area
            best_overlap = max(best_overlap, overlap_ratio)
            if overlap_ratio >= self._min_overlap:
                triggered = True

        stop_state = self._debouncer.update(triggered)
        return stop_state, best_overlap

    def _draw_roi_overlay(self, canvas: np.ndarray, stop_state: bool):
        """Draw the ROI rectangle (green = clear, red = protective stop
        triggered) directly onto the annotated detections visualisation."""
        h, w = canvas.shape[:2]
        ix1, iy1, ix2, iy2 = _roi_pixel_bounds(self._roi, w, h)
        color = (0, 0, 255) if stop_state else (0, 200, 0)  # BGR
        cv2.rectangle(canvas, (ix1, iy1), (ix2, iy2), color, 3)
        label = "PROTECTIVE STOP" if stop_state else "clear"
        cv2.putText(
            canvas, label, (ix1 + 4, max(iy1 - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )

    def _publish_detections(self, detections, stamp, frame_id):
        msg = Detection2DArray()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        for x1, y1, x2, y2, conf, lbl in detections:
            det = Detection2D()
            det.header.stamp = stamp
            det.header.frame_id = frame_id
            label = self._label_text(lbl)

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = label
            hyp.hypothesis.score = conf
            det.results.append(hyp)

            bbox = BoundingBox2D()
            bbox.center = Pose2D()
            bbox.center.position.x = (x1 + x2) / 2.0
            bbox.center.position.y = (y1 + y2) / 2.0
            bbox.size_x = max(x2 - x1, 0.0)
            bbox.size_y = max(y2 - y1, 0.0)
            det.bbox = bbox
            det.id = label

            msg.detections.append(det)
        self._det_pub.publish(msg)

    def _draw_and_publish_viz(self, bgr_frame, detections, stop_state, stamp, frame_id):
        canvas = bgr_frame.copy()
        for x1, y1, x2, y2, conf, lbl in detections:
            color = _PALETTE[lbl % len(_PALETTE)]
            label = self._label_text(lbl)
            p1 = (int(x1), int(y1))
            p2 = (int(x2), int(y2))
            cv2.rectangle(canvas, p1, p2, color, 2)
            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                canvas, (p1[0], max(p1[1] - th - 4, 0)), (p1[0] + tw + 2, p1[1]), color, -1
            )
            cv2.putText(
                canvas, text, (p1[0] + 1, max(p1[1] - 3, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )

        self._draw_roi_overlay(canvas, stop_state)

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        vis = Image()
        vis.header.stamp = stamp
        vis.header.frame_id = frame_id
        vis.height = h
        vis.width = w
        vis.encoding = "rgb8"
        vis.is_bigendian = False
        vis.step = w * 3
        vis.data = array.array("B", np.ascontiguousarray(rgb).tobytes())
        self._pub.publish(vis)

    def _image_cb(self, msg: Image):
        if self._throttled():
            return
        try:
            bgr = self._image_to_bgr(msg)
        except ValueError as exc:
            self.get_logger().warn(str(exc), throttle_duration_sec=5.0)
            return
        stamp = msg.header.stamp if msg.header.stamp.sec else self.get_clock().now().to_msg()
        frame_id = msg.header.frame_id or "camera"
        self._process(bgr, stamp, frame_id)

    def _compressed_cb(self, msg: CompressedImage):
        if self._throttled():
            return
        buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            self.get_logger().warn(
                "Failed to decode compressed image", throttle_duration_sec=5.0
            )
            return
        stamp = msg.header.stamp if msg.header.stamp.sec else self.get_clock().now().to_msg()
        frame_id = msg.header.frame_id or "camera"
        self._process(bgr, stamp, frame_id)

    def _throttled(self) -> bool:
        # Optional inference throttle so a slow pipeline doesn't build up a
        # backlog (the keep-last-1 QoS already drops most of it).
        if self._min_period > 0.0:
            now = time.monotonic()
            if now - self._last_infer_t < self._min_period:
                return True
            self._last_infer_t = now
        return False

    def _process(self, bgr, stamp, frame_id):
        h, w = bgr.shape[:2]
        t_infer0 = time.monotonic()
        detections = self._predict(bgr)
        infer_ms = (time.monotonic() - t_infer0) * 1000.0

        # Safety is authoritative -> ALWAYS publish the protective-stop state.
        stop_state, best_overlap = self._evaluate_stop(detections, w, h)
        self._stop_pub.publish(Bool(data=stop_state))

        # The detections + annotated viz are only for consumers (rqt / RViz /
        # other nodes). Building the ~920 KB rgb8 viz and pushing it over DDS is
        # pure overhead when nobody is subscribed -> skip it then. This keeps the
        # transport/memory load down (which is what stalls the camera feed) while
        # the arm is just running headless.
        if self._det_pub.get_subscription_count() > 0:
            self._publish_detections(detections, stamp, frame_id)
        if self._pub.get_subscription_count() > 0:
            self._draw_and_publish_viz(bgr, detections, stop_state, stamp, frame_id)

        self._frame_count += 1
        if self._frame_count % 30 == 0:
            self.get_logger().info(
                f"NanoOWL inference running — frame {self._frame_count}, "
                f"{len(detections)} detection(s), overlap={best_overlap:.2f}, "
                f"stop={stop_state}, infer={infer_ms:.0f}ms "
                f"(~{1000.0 / max(infer_ms, 1e-6):.1f} fps)"
            )


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = OwlDetectNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()

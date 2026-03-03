"""
YOLO Dual-Model Accuracy Comparator  (No Ground Truth Required)
===============================================================
Runs two trained YOLO models on the same test video, compares them
across proxy-accuracy metrics, and optionally writes an annotated
side-by-side output video.

Output video layout
───────────────────
┌──────────────── Header A (50 px) ─────┬──────────── Header B (50 px) ──────┐
│ Model A: <name>  Det:N  Conf:0.XX FPS │ Model B: <name>  Det:N  Conf:0.XX  │
├───────────────────────────────────────┼────────────────────────────────────┤
│                                       │                                    │
│   Raw frame + custom bounding boxes   │  Raw frame + custom bounding boxes │
│   Each box shows:                     │  (same color palette per class)    │
│     • class name                      │                                    │
│     • class ID                        │                                    │
│     • confidence %                    │                                    │
│     • box W×H px                      │                                    │
│                                       │                                    │
├───────────────────────────────── Info bar (90 px) ─────────────────────────┤
│ Frame: N/Total  Time: MM:SS/MM:SS  Agreement: 0.XX  Overlap: 0.XX          │
│ [A]  FPS: XX.X  AvgConf: 0.XXX  TotalDet: XXXX  Hi-Conf%: XX.X%           │
│ [B]  FPS: XX.X  AvgConf: 0.XXX  TotalDet: XXXX  Hi-Conf%: XX.X%           │
└────────────────────────────────────────────────────────────────────────────┘

Proxy-accuracy metrics (no ground truth needed)
───────────────────────────────────────────────
 1. Average detection confidence
 2. Median / std-dev of confidence
 3. High-confidence ratio  (≥ 0.70)
 4. Low-confidence  ratio  (< 0.40)   ← lower is better
 5. Temporal stability     (frame-to-frame IoU continuity)
 6. Detection coverage     (% frames with ≥1 detection)
 7. Avg detections per frame + count std-dev
 8. NMS quality score      (1 − avg intra-frame IoU)
 9. Inference speed        (FPS)
10. Composite proxy-accuracy (weighted, 0–100)
11. Inter-model agreement  (cross-model IoU matching)
12. Per-class statistics

Usage
─────
  python yolo_model_comparator.py test.mp4 modelA.pt modelB.pt
  python yolo_model_comparator.py test.mp4 yolov8n.pt yolov8s.pt \\
      --label-a YOLOv8n --label-b YOLOv8s \\
      --save-video output.mp4 --show --chart --save-json results.json
"""

import time, json, argparse
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette  (BGR, 20 visually distinct colours, cycled by class-id)
# ─────────────────────────────────────────────────────────────────────────────

_PALETTE = [
    (56,  56,  255), (151, 157, 255), (31,  112, 255), (29,  178, 255),
    (49,  210, 207), (10,  249,  72), (23,  204, 146), (134, 219,  61),
    (52,  147,  26), (187, 212,   0), (168, 153,  44), (255, 194,   0),
    (147,  69,  52), (255, 115, 100), (236,  24,   0), (255,  56, 132),
    (133,   0,  82), (255,  56, 203), (200, 149, 255), (199,  55, 255),
]

def class_color(cls_id: int):
    return _PALETTE[int(cls_id) % len(_PALETTE)]


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

_FONT      = cv2.FONT_HERSHEY_SIMPLEX
_FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

def _text_size(text, scale, thickness=1):
    (w, h), baseline = cv2.getTextSize(text, _FONT, scale, thickness)
    return w, h, baseline


def _contrasting(bgr):
    """Return black or white depending on background brightness."""
    b, g, r = bgr
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if lum > 140 else (255, 255, 255)


def draw_detections(frame: np.ndarray, results, model_label: str = "Model A") -> np.ndarray:
    """
    Draw bounding boxes on *frame* (handles both OBB and standard boxes) with:
      • Coloured border  (colour keyed to class ID)
      • Filled label tab above the box:
            Line 1 (larger): '<classname>  <conf>%'
            Line 2 (smaller): 'ID:<n>  <W>×<H> px'
    Returns an annotated copy; does NOT modify the original.
    """
    out = frame.copy()
    result = results[0]
    
    # Get boxes (OBB or standard)
    if result.obb is not None and len(result.obb) > 0:
        boxes_obj = result.obb
        box_type = 'obb'
    elif result.boxes is not None and len(result.boxes) > 0:
        boxes_obj = result.boxes
        box_type = 'standard'
    else:
        return out

    names = result.names

    if box_type == 'obb':
        # Handle Oriented Bounding Boxes
        if hasattr(boxes_obj, 'xyxyxyxy'):
            corners = boxes_obj.xyxyxyxy.cpu().numpy()
            cls_ids = boxes_obj.cls.cpu().numpy().astype(int)
            confs = boxes_obj.conf.cpu().numpy()
            xyxy = boxes_obj.xyxy.cpu().numpy()  # For bounding box
            
            for corner_pts, cls_id, conf, box_coords in zip(corners, cls_ids, confs, xyxy):
                x1, y1, x2, y2 = map(int, box_coords)
                cls_name = names[int(cls_id)]
                color = class_color(int(cls_id))
                bw, bh = x2 - x1, y2 - y1
                
                # Draw rotated rectangle outline
                pts = corner_pts.astype(np.int32)
                box_thick = max(2, min(4, bw // 80))
                cv2.polylines(out, [pts], True, color, box_thick, cv2.LINE_AA)
                
                # Draw label tab (use bounding box coords for positioning)
                line1 = f"{cls_name}  {conf * 100:.1f}%"
                line2 = f"ID:{cls_id}  {bw}x{bh}px"
                fs1, fs2 = 0.52, 0.38
                tw1, th1, _ = _text_size(line1, fs1)
                tw2, th2, _ = _text_size(line2, fs2)
                pad = 4
                tab_w = max(tw1, tw2) + pad * 2
                tab_h = th1 + th2 + pad * 3
                
                tab_y = y1 - tab_h if y1 - tab_h >= 0 else y1
                
                overlay = out.copy()
                cv2.rectangle(overlay, (x1, tab_y), (x1 + tab_w, tab_y + tab_h), color, -1)
                cv2.addWeighted(overlay, 0.85, out, 0.15, 0, out)
                
                tc = _contrasting(color)
                cv2.putText(out, line1, (x1 + pad, tab_y + th1 + pad),
                           _FONT, fs1, tc, 1, cv2.LINE_AA)
                cv2.putText(out, line2, (x1 + pad, tab_y + th1 + th2 + pad * 2),
                           _FONT, fs2, tc, 1, cv2.LINE_AA)
    else:
        # Handle standard bounding boxes
        for box in boxes_obj:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = names[cls_id]
            color = class_color(cls_id)
            bw, bh = x2 - x1, y2 - y1

            # ── bounding box ──────────────────────────────────────────────────
            box_thick = max(2, min(4, bw // 80))
            cv2.rectangle(out, (x1, y1), (x2, y2), color, box_thick)

            # ── label text ────────────────────────────────────────────────────
            line1 = f"{cls_name}  {conf * 100:.1f}%"
            line2 = f"ID:{cls_id}  {bw}x{bh}px"
            fs1, fs2 = 0.52, 0.38
            tw1, th1, _ = _text_size(line1, fs1)
            tw2, th2, _ = _text_size(line2, fs2)

            pad = 4
            tab_w = max(tw1, tw2) + pad * 2
            tab_h = th1 + th2 + pad * 3

            # Place tab above box; if no space, overlay inside at top
            tab_y = y1 - tab_h if y1 - tab_h >= 0 else y1

            # Filled background (slightly transparent via overlay)
            overlay = out.copy()
            cv2.rectangle(overlay,
                          (x1, tab_y), (x1 + tab_w, tab_y + tab_h),
                          color, -1)
            cv2.addWeighted(overlay, 0.85, out, 0.15, 0, out)

            tc = _contrasting(color)
            cv2.putText(out, line1,
                        (x1 + pad, tab_y + th1 + pad),
                        _FONT, fs1, tc, 1, cv2.LINE_AA)
            cv2.putText(out, line2,
                        (x1 + pad, tab_y + th1 + th2 + pad * 2),
                        _FONT, fs2, tc, 1, cv2.LINE_AA)

    return out


def draw_dual_detections(frame: np.ndarray, results_a, results_b, model_label_a: str = "Model A", model_label_b: str = "Model B") -> np.ndarray:
    """
    Overlay detections from both models on the same frame.
    Model A: solid lines, Model B: dashed lines for visual distinction.
    """
    out = frame.copy()
    
    # Color schemes for distinction
    def get_styled_color(cls_id: int, is_model_b: bool = False):
        base_color = class_color(int(cls_id))
        if is_model_b:
            # Lighten the color for model B for distinction
            return tuple(min(255, int(c * 0.7)) for c in base_color)
        return base_color
    
    # Draw Model A (solid lines)
    out = _draw_detections_styled(out, results_a, is_model_b=False)
    
    # Draw Model B (different line style)
    out = _draw_detections_styled(out, results_b, is_model_b=True)
    
    return out


def _draw_detections_styled(frame: np.ndarray, results, is_model_b: bool = False) -> np.ndarray:
    """Helper to draw detections with optional visual distinction."""
    out = frame.copy()
    result = results[0]
    
    # Get boxes (OBB or standard)
    if result.obb is not None and len(result.obb) > 0:
        boxes_obj = result.obb
        box_type = 'obb'
    elif result.boxes is not None and len(result.boxes) > 0:
        boxes_obj = result.boxes
        box_type = 'standard'
    else:
        return out

    names = result.names
    
    def get_styled_color(cls_id: int):
        base_color = class_color(int(cls_id))
        if is_model_b:
            # Darken color for model B
            return tuple(max(0, int(c * 0.6)) for c in base_color)
        return base_color

    if box_type == 'obb':
        if hasattr(boxes_obj, 'xyxyxyxy'):
            corners = boxes_obj.xyxyxyxy.cpu().numpy()
            cls_ids = boxes_obj.cls.cpu().numpy().astype(int)
            confs = boxes_obj.conf.cpu().numpy()
            xyxy = boxes_obj.xyxy.cpu().numpy()
            
            for corner_pts, cls_id, conf, box_coords in zip(corners, cls_ids, confs, xyxy):
                x1, y1, x2, y2 = map(int, box_coords)
                cls_name = names[int(cls_id)]
                color = get_styled_color(int(cls_id))
                bw, bh = x2 - x1, y2 - y1
                
                pts = corner_pts.astype(np.int32)
                box_thick = max(2, min(4, bw // 80))
                
                # Draw with different style
                if is_model_b:
                    # Dashed line style (simulate with segments)
                    for i in range(len(pts)):
                        p1 = pts[i]
                        p2 = pts[(i + 1) % len(pts)]
                        cv2.line(out, tuple(p1), tuple(p2), color, box_thick, cv2.LINE_AA)
                else:
                    cv2.polylines(out, [pts], True, color, box_thick, cv2.LINE_AA)
                
                # Minimal label for dual view
                text = f"{cls_name}"
                cv2.putText(out, text, tuple(pts[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX,
                           0.4, color, 1, cv2.LINE_AA)
    else:
        for box in boxes_obj:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = names[cls_id]
            color = get_styled_color(cls_id)
            
            box_thick = max(1, min(3, (x2 - x1) // 100))
            
            if is_model_b:
                # Draw dashed rectangle for model B
                for i in range(x1, x2, 10):
                    cv2.line(out, (i, y1), (min(i + 5, x2), y1), color, box_thick, cv2.LINE_AA)
                    cv2.line(out, (i, y2), (min(i + 5, x2), y2), color, box_thick, cv2.LINE_AA)
                for i in range(y1, y2, 10):
                    cv2.line(out, (x1, i), (x1, min(i + 5, y2)), color, box_thick, cv2.LINE_AA)
                    cv2.line(out, (x2, i), (x2, min(i + 5, y2)), color, box_thick, cv2.LINE_AA)
            else:
                cv2.rectangle(out, (x1, y1), (x2, y2), color, box_thick)
            
            # Minimal label
            text = f"{cls_name}"
            cv2.putText(out, text, (x1, max(y1 - 3, 0)), cv2.FONT_HERSHEY_SIMPLEX,
                       0.4, color, 1, cv2.LINE_AA)

    return out


def make_header_band(label: str, det_count: int,
                     frame_conf: float, run_avg_conf: float,
                     fps: float, width: int,
                     height: int = 50,
                     bg_color=(30, 30, 30)) -> np.ndarray:
    """
    Creates a dark horizontal band for one model with live stats:
      Model A: YOLOv8n  |  Dets: 5  |  Frame conf: 0.823  |  Run avg: 0.791  |  FPS: 87.2
    """
    band = np.full((height, width, 3), bg_color, dtype=np.uint8)
    text = (f"  {label}"
            f"   |   Dets: {det_count}"
            f"   |   Frame conf: {frame_conf:.3f}"
            f"   |   Run avg: {run_avg_conf:.3f}"
            f"   |   FPS: {fps:.1f}")
    tw, th, _ = _text_size(text, 0.52, 1)
    y = (height + th) // 2
    cv2.putText(band, text, (4, y), _FONT, 0.52, (200, 230, 255), 1, cv2.LINE_AA)
    return band


def make_info_bar(total_width: int,
                  frame_num: int, total_frames: int, video_fps: float,
                  label_a: str, label_b: str,
                  stats_a: dict, stats_b: dict,
                  agreement: float, overlap: float,
                  height: int = 90,
                  bg_color=(20, 20, 20)) -> np.ndarray:
    """
    Three-row info bar spanning the full output width:
      Row 1: frame counter, timestamp, cross-model agreement & overlap
      Row 2: running stats for Model A
      Row 3: running stats for Model B
    """
    bar = np.full((height, total_width, 3), bg_color, dtype=np.uint8)
    row_h = height // 3

    elapsed_s = frame_num / max(video_fps, 1)
    total_s   = total_frames / max(video_fps, 1)

    def fmt_time(s):
        m, sec = divmod(int(s), 60)
        return f"{m:02d}:{sec:02d}"

    # Row 1 — frame / time / agreement
    r1 = (f"  Frame: {frame_num}/{total_frames}"
          f"   |   Time: {fmt_time(elapsed_s)}/{fmt_time(total_s)}"
          f"   |   Agreement: {agreement:.3f}"
          f"   |   Det overlap: {overlap:.3f}")
    _, th1, _ = _text_size(r1, 0.48)
    cv2.putText(bar, r1,
                (4, row_h // 2 + th1 // 2),
                _FONT, 0.48, (160, 220, 160), 1, cv2.LINE_AA)

    # Row 2 — Model A
    sa = stats_a
    r2 = (f"  [{label_a}]"
          f"  FPS: {sa.get('fps', 0):.1f}"
          f"  |  AvgConf: {sa.get('avg_conf', 0):.3f}"
          f"  |  TotalDet: {sa.get('total_det', 0)}"
          f"  |  Hi-Conf%: {sa.get('high_conf', 0)*100:.1f}%"
          f"  |  Stability: {sa.get('stability', 0):.3f}")
    _, th2, _ = _text_size(r2, 0.46)
    cv2.putText(bar, r2,
                (4, row_h + row_h // 2 + th2 // 2),
                _FONT, 0.46, (100, 180, 255), 1, cv2.LINE_AA)

    # Row 3 — Model B
    sb = stats_b
    r3 = (f"  [{label_b}]"
          f"  FPS: {sb.get('fps', 0):.1f}"
          f"  |  AvgConf: {sb.get('avg_conf', 0):.3f}"
          f"  |  TotalDet: {sb.get('total_det', 0)}"
          f"  |  Hi-Conf%: {sb.get('high_conf', 0)*100:.1f}%"
          f"  |  Stability: {sb.get('stability', 0):.3f}")
    _, th3, _ = _text_size(r3, 0.46)
    cv2.putText(bar, r3,
                (4, 2 * row_h + row_h // 2 + th3 // 2),
                _FONT, 0.46, (255, 200, 100), 1, cv2.LINE_AA)

    return bar


def build_output_frame(frame_combined: np.ndarray,
                       header_dual: np.ndarray,
                       info_bar: np.ndarray) -> np.ndarray:
    """Assemble the full output frame from its parts (single frame mode)."""
    return np.vstack([header_dual, frame_combined, info_bar])


def make_dual_header_band(label_a: str, label_b: str,
                          det_a: int, det_b: int,
                          conf_a: float, conf_b: float,
                          run_avg_a: float, run_avg_b: float,
                          fps_a: float, fps_b: float,
                          width: int,
                          height: int = 50,
                          bg_color=(30, 30, 30)) -> np.ndarray:
    """Create header showing both models' stats side by side."""
    band = np.full((height, width, 3), bg_color, dtype=np.uint8)
    text = (f"  [{label_a}] Dets:{det_a} Conf:{conf_a:.3f} Avg:{run_avg_a:.3f} FPS:{fps_a:.1f}"
            f"  |  "
            f"[{label_b}] Dets:{det_b} Conf:{conf_b:.3f} Avg:{run_avg_b:.3f} FPS:{fps_b:.1f}")
    tw, th, _ = _text_size(text, 0.45, 1)
    y = (height + th) // 2
    cv2.putText(band, text, (4, y), _FONT, 0.45, (200, 230, 255), 1, cv2.LINE_AA)
    return band


# ─────────────────────────────────────────────────────────────────────────────
# IoU helpers
# ─────────────────────────────────────────────────────────────────────────────

def box_iou(b1, b2):
    xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def match_boxes(boxes_a, boxes_b, iou_thr=0.4):
    if not boxes_a or not boxes_b:
        return 0, len(boxes_a) + len(boxes_b)
    matched, used_b = 0, set()
    for a in boxes_a:
        best, best_j = iou_thr, -1
        for j, b in enumerate(boxes_b):
            if j in used_b:
                continue
            v = box_iou(a, b)
            if v > best:
                best, best_j = v, j
        if best_j >= 0:
            matched += 1
            used_b.add(best_j)
    return matched, len(boxes_a) + len(boxes_b) - matched


# ─────────────────────────────────────────────────────────────────────────────
# Per-model metric collector
# ─────────────────────────────────────────────────────────────────────────────

class ModelMetricCollector:

    def __init__(self, model_path: str, conf: float, label: str):
        print(f"  Loading {label}: {model_path} …")
        self.model   = YOLO(model_path)
        self.conf    = conf
        self.label   = label
        self._reset()

    def _reset(self):
        self.confidences     = []
        self.per_class_conf  = defaultdict(list)
        self.frame_counts    = []
        self.temporal_stab   = []
        self.nms_overlaps    = []
        self.prev_boxes      = []
        self.infer_times     = []

    # ── inference ──────────────────────────────────────────────────────────

    def infer(self, frame):
        t0 = time.perf_counter()
        results = self.model(frame, conf=self.conf, verbose=False)
        self.infer_times.append(time.perf_counter() - t0)
        return results

    # ── per-frame update ────────────────────────────────────────────────────

    def _get_boxes(self, result):
        """Get boxes from result, agnostic of OBB vs standard boxes."""
        if result.obb is not None and len(result.obb) > 0:
            return result.obb, 'obb'
        elif result.boxes is not None and len(result.boxes) > 0:
            return result.boxes, 'standard'
        return None, None

    def update(self, results):
        result = results[0]
        boxes_obj, box_type = self._get_boxes(result)
        
        if boxes_obj is None:
            self.frame_counts.append(0)
            if self.prev_boxes:
                self.temporal_stab.append(0.0)
            self.prev_boxes = []
            return [], []

        confs   = boxes_obj.conf.cpu().numpy()
        cls_ids = boxes_obj.cls.cpu().numpy().astype(int)
        xyxy    = boxes_obj.xyxy.cpu().numpy()

        self.confidences.extend(confs.tolist())
        for c, cid in zip(confs, cls_ids):
            self.per_class_conf[self.model.names[cid]].append(float(c))

        self.frame_counts.append(len(confs))

        if self.prev_boxes:
            stab = [max((box_iou(cur, p) for p in self.prev_boxes), default=0.0)
                    for cur in xyxy]
            self.temporal_stab.append(float(np.mean(stab)))

        if len(xyxy) > 1:
            pairs = [box_iou(xyxy[i], xyxy[j])
                     for i in range(len(xyxy))
                     for j in range(i + 1, len(xyxy))]
            self.nms_overlaps.append(float(np.mean(pairs)))
        else:
            self.nms_overlaps.append(0.0)

        self.prev_boxes = xyxy.tolist()
        return xyxy.tolist(), confs.tolist()

    # ── running snapshot (called every frame for the info bar) ─────────────

    def running_stats(self) -> dict:
        """Cheap snapshot of running averages for the on-screen info bar."""
        confs = self.confidences
        return {
            "fps":        1.0 / np.mean(self.infer_times) if self.infer_times else 0,
            "avg_conf":   float(np.mean(confs))                    if confs else 0.0,
            "total_det":  len(confs),
            "high_conf":  float(np.mean(np.array(confs) >= 0.70))  if confs else 0.0,
            "stability":  float(np.mean(self.temporal_stab)) if self.temporal_stab else 0.0,
        }

    # ── frame-level snapshot (for header band) ─────────────────────────────

    def frame_stats(self, results) -> tuple:
        """Return (det_count, frame_conf) for the current frame."""
        result = results[0]
        boxes_obj, _ = self._get_boxes(result)
        if boxes_obj is None:
            return 0, 0.0
        confs = boxes_obj.conf.cpu().numpy()
        return int(len(confs)), float(np.mean(confs))

    # ── final aggregation ──────────────────────────────────────────────────

    def compute(self) -> dict:
        m = {}
        if self.confidences:
            arr = np.array(self.confidences)
            m["avg_confidence"]    = float(np.mean(arr))
            m["median_confidence"] = float(np.median(arr))
            m["conf_std_dev"]      = float(np.std(arr))
            m["high_conf_ratio"]   = float(np.mean(arr >= 0.70))
            m["low_conf_ratio"]    = float(np.mean(arr < 0.40))
            m["total_detections"]  = int(len(arr))
        else:
            m.update(avg_confidence=0.0, median_confidence=0.0,
                     conf_std_dev=0.0, high_conf_ratio=0.0,
                     low_conf_ratio=1.0, total_detections=0)

        m["temporal_stability"] = (float(np.mean(self.temporal_stab))
                                   if self.temporal_stab else 0.0)

        if self.frame_counts:
            cnt = np.array(self.frame_counts)
            m["avg_detections_per_frame"] = float(np.mean(cnt))
            m["detection_count_std"]      = float(np.std(cnt))
            m["frames_with_detections"]   = int(np.sum(cnt > 0))
            m["total_frames"]             = int(len(cnt))
            m["detection_coverage"]       = float(np.mean(cnt > 0))
        else:
            m.update(avg_detections_per_frame=0.0, detection_count_std=0.0,
                     frames_with_detections=0, total_frames=0,
                     detection_coverage=0.0)

        if self.nms_overlaps:
            m["avg_intra_iou"]     = float(np.mean(self.nms_overlaps))
            m["nms_quality_score"] = float(1.0 - np.mean(self.nms_overlaps))
        else:
            m["avg_intra_iou"]     = 0.0
            m["nms_quality_score"] = 1.0

        if self.infer_times:
            m["avg_ms_per_frame"] = float(np.mean(self.infer_times) * 1000)
            m["inference_fps"]    = float(1.0 / np.mean(self.infer_times))
        else:
            m["avg_ms_per_frame"] = 0.0
            m["inference_fps"]    = 0.0

        per_cls = {}
        for cls, clist in self.per_class_conf.items():
            arr = np.array(clist)
            per_cls[cls] = dict(count=int(len(arr)),
                                avg_conf=round(float(np.mean(arr)), 4),
                                min_conf=round(float(np.min(arr)),  4),
                                max_conf=round(float(np.max(arr)),  4))
        m["per_class_stats"] = per_cls

        composite = (0.35 * m["avg_confidence"]
                   + 0.30 * m["temporal_stability"]
                   + 0.15 * m["nms_quality_score"]
                   + 0.15 * m["high_conf_ratio"]
                   + 0.05 * m["detection_coverage"]) * 100
        m["composite_score"] = round(composite, 2)

        return m


# ─────────────────────────────────────────────────────────────────────────────
# Main comparator
# ─────────────────────────────────────────────────────────────────────────────

HEADER_H = 50   # px — header band height per model column
INFO_H   = 90   # px — bottom info bar height

class YOLOModelComparator:

    def __init__(self, model_a: str, model_b: str,
                 conf: float = 0.25, iou_thr: float = 0.4,
                 label_a: str = "Model A", label_b: str = "Model B"):
        self.label_a  = label_a
        self.label_b  = label_b
        self.iou_thr  = iou_thr
        self.col_a    = ModelMetricCollector(model_a, conf, label_a)
        self.col_b    = ModelMetricCollector(model_b, conf, label_b)
        self.frame_agreements  = []
        self.detection_overlaps = []

    # ── agreement helper ────────────────────────────────────────────────────

    def _update_agreement(self, boxes_a, boxes_b):
        if not boxes_a and not boxes_b:
            self.frame_agreements.append(1.0)
            return
        if not boxes_a or not boxes_b:
            self.frame_agreements.append(0.0)
            self.detection_overlaps.append(0.0)
            return
        matched, unmatched = match_boxes(boxes_a, boxes_b, self.iou_thr)
        total = matched + unmatched
        ov = matched / total if total > 0 else 0.0
        self.frame_agreements.append(ov)
        self.detection_overlaps.append(ov)

    # ── main loop ───────────────────────────────────────────────────────────

    def run(self, video_path: str,
            show: bool = False,
            save_video: str = None) -> dict:

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open: {video_path}")

        total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_src = cap.get(cv2.CAP_PROP_FPS)
        W       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Output video dimensions
        out_w = W
        out_h = H + HEADER_H + INFO_H

        print(f"\nVideo  : {video_path}")
        print(f"Frames : {total}   FPS: {fps_src:.1f}   Resolution: {W}×{H}")
        print(f"Output : {out_w}×{out_h}  (dual model overlay + overlays)\n")

        # ── VideoWriter ───────────────────────────────────────────────────
        writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(save_video, fourcc, fps_src,
                                     (out_w, out_h))
            if not writer.isOpened():
                print(f"[warn] Could not open VideoWriter for {save_video!r}. "
                      "Trying XVID …")
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                out_avi = save_video.rsplit(".", 1)[0] + ".avi"
                writer  = cv2.VideoWriter(out_avi, fourcc, fps_src,
                                          (out_w, out_h))
                save_video = out_avi

        frame_num = 0
        run_agreement = 0.0
        run_overlap   = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # ── inference ─────────────────────────────────────────────────
            res_a = self.col_a.infer(frame)
            res_b = self.col_b.infer(frame)

            boxes_a, _ = self.col_a.update(res_a)
            boxes_b, _ = self.col_b.update(res_b)
            self._update_agreement(boxes_a, boxes_b)

            frame_num += 1

            # Running averages for display
            run_agreement = np.mean(self.frame_agreements)
            run_overlap   = (np.mean(self.detection_overlaps)
                             if self.detection_overlaps else 0.0)

            # ── build output frame ─────────────────────────────────────────
            if show or writer:
                # Overlay detections from both models on single frame
                combined = draw_dual_detections(frame, res_a, res_b, 
                                                self.label_a, self.label_b)

                # Per-frame stats
                det_a, fc_a = self.col_a.frame_stats(res_a)
                det_b, fc_b = self.col_b.frame_stats(res_b)
                rs_a = self.col_a.running_stats()
                rs_b = self.col_b.running_stats()

                hdr_dual = make_dual_header_band(
                    self.label_a, self.label_b,
                    det_a, det_b, fc_a, fc_b,
                    rs_a["avg_conf"], rs_b["avg_conf"],
                    rs_a["fps"], rs_b["fps"], W, HEADER_H)

                ibar = make_info_bar(
                    out_w, frame_num, total, fps_src,
                    self.label_a, self.label_b,
                    rs_a, rs_b,
                    run_agreement, run_overlap, INFO_H)

                output_frame = build_output_frame(combined, hdr_dual, ibar)

                if writer:
                    writer.write(output_frame)

                if show:
                    # Scale for display if frame is too wide
                    disp = output_frame
                    max_disp_w = 1600
                    if disp.shape[1] > max_disp_w:
                        scale = max_disp_w / disp.shape[1]
                        disp = cv2.resize(disp,
                                         (max_disp_w,
                                          int(disp.shape[0] * scale)))
                    cv2.imshow(f"{self.label_a}  ∩  {self.label_b}  (Overlay)", disp)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            if frame_num % 50 == 0:
                print(f"  [{frame_num:>5}/{total}] "
                      f"agree={run_agreement:.3f}  "
                      f"conf_A={self.col_a.running_stats()['avg_conf']:.3f}  "
                      f"conf_B={self.col_b.running_stats()['avg_conf']:.3f}")

        cap.release()
        if writer:
            writer.release()
            print(f"\n[video] Saved → {save_video}")
        if show:
            cv2.destroyAllWindows()

        print(f"\nDone — {frame_num} frames processed.\n")
        return self.compile_comparison()

    # ── compile ─────────────────────────────────────────────────────────────

    def compile_comparison(self) -> dict:
        ma = self.col_a.compute()
        mb = self.col_b.compute()
        cross = {
            "inter_model_agreement": (
                float(np.mean(self.frame_agreements))
                if self.frame_agreements else 0.0),
            "detection_overlap_score": (
                float(np.mean(self.detection_overlaps))
                if self.detection_overlaps else 0.0),
        }
        return {self.label_a: ma, self.label_b: mb,
                "cross_model": cross,
                "labels": (self.label_a, self.label_b)}


# ─────────────────────────────────────────────────────────────────────────────
# Terminal report
# ─────────────────────────────────────────────────────────────────────────────

def _delta(va, vb, higher_is_better=True):
    if va == vb:
        return "Tie", "  0.0%"
    pct = abs(va - vb) / (abs(vb) + 1e-9) * 100
    if higher_is_better:
        winner = "A" if va > vb else "B"
    else:
        winner = "A" if va < vb else "B"
    sign = "+" if winner == "A" else "-"
    return winner, f"{sign}{pct:.1f}%"


def print_report(cmp: dict):
    la, lb = cmp["labels"]
    ma, mb = cmp[la], cmp[lb]
    cross  = cmp["cross_model"]
    W      = 75

    def sep(c="─"): print(c * W)
    def row(label, va, vb, fmt=".4f", hib=True):
        winner, delta = _delta(va, vb, hib)
        ma_ = " ◀" if winner == "A" else ""
        mb_ = " ◀" if winner == "B" else ""
        print(f"  {label:<32} {f'{va:{fmt}}{ma_}':<18} "
              f"{f'{vb:{fmt}}{mb_}':<18} {delta:>7}")

    sep("═")
    print("  YOLO MODEL COMPARISON REPORT")
    sep("═")
    print(f"  {'Metric':<32} {la:<18} {lb:<18} {'Δ (A vs B)':>7}")
    sep()

    print("\n── Confidence ──")
    row("Avg confidence",          ma["avg_confidence"],    mb["avg_confidence"])
    row("Median confidence",       ma["median_confidence"], mb["median_confidence"])
    row("Confidence std-dev",      ma["conf_std_dev"],      mb["conf_std_dev"],   hib=False)
    row("High-conf ratio (≥0.70)", ma["high_conf_ratio"],   mb["high_conf_ratio"], fmt=".2%")
    row("Low-conf  ratio (<0.40)", ma["low_conf_ratio"],    mb["low_conf_ratio"],  fmt=".2%", hib=False)

    print("\n── Temporal Stability ──")
    row("Stability score (0–1)",   ma["temporal_stability"], mb["temporal_stability"])

    print("\n── Detection Coverage ──")
    row("Coverage (frames w/ det)",ma["detection_coverage"],mb["detection_coverage"], fmt=".2%")
    row("Avg detections / frame",  ma["avg_detections_per_frame"], mb["avg_detections_per_frame"])
    row("Detection count std-dev", ma["detection_count_std"],      mb["detection_count_std"], hib=False)

    print("\n── NMS / Box Quality ──")
    row("Avg intra-frame IoU",     ma["avg_intra_iou"],     mb["avg_intra_iou"],     hib=False)
    row("NMS quality score (0–1)", ma["nms_quality_score"], mb["nms_quality_score"])

    print("\n── Speed ──")
    row("Inference FPS",           ma["inference_fps"],    mb["inference_fps"],    fmt=".1f")
    row("Avg ms / frame",          ma["avg_ms_per_frame"], mb["avg_ms_per_frame"], fmt=".1f", hib=False)

    print("\n── Totals ──")
    row("Total detections",        ma["total_detections"], mb["total_detections"], fmt="d")
    row("Frames processed",        ma["total_frames"],     mb["total_frames"],     fmt="d")
    sep()

    print("\n── Cross-Model Agreement ──")
    print(f"  {'Inter-model agreement':<32} {cross['inter_model_agreement']:.4f}"
          f"   (fraction of frames where both models agree)")
    print(f"  {'Detection overlap score':<32} {cross['detection_overlap_score']:.4f}"
          f"   (fraction of matched boxes across both models)")
    sep()

    sa, sb = ma["composite_score"], mb["composite_score"]
    grade  = lambda s: ("Excellent" if s>=80 else "Good" if s>=65
                        else "Fair" if s>=50 else "Poor")
    print(f"\n  COMPOSITE PROXY-ACCURACY SCORES")
    print(f"  {la:<18}: {sa:>6.2f} / 100  [{grade(sa)}]")
    print(f"  {lb:<18}: {sb:>6.2f} / 100  [{grade(sb)}]")

    overall = la if sa > sb else (lb if sb > sa else "Tie")
    diff    = abs(sa - sb)
    if overall != "Tie":
        print(f"\n  >>> OVERALL WINNER : {overall}  "
              f"(+{diff:.2f} pts / +{diff/(sb+1e-9)*100:.1f}%)")
    else:
        print("\n  >>> RESULT : TIE")

    sep("═")

    # Per-class
    all_cls = sorted(set(ma["per_class_stats"]) | set(mb["per_class_stats"]))
    if all_cls:
        print(f"\n{'═'*W}")
        print("  PER-CLASS COMPARISON")
        sep()
        print(f"  {'Class':<20} {'Cnt-A':>6}  {'AvgConf-A':>9}"
              f"  {'Cnt-B':>6}  {'AvgConf-B':>9}  {'Better':>10}")
        sep()
        for cls in all_cls:
            sa_c = ma["per_class_stats"].get(cls, {})
            sb_c = mb["per_class_stats"].get(cls, {})
            ca, cb = sa_c.get("count", 0), sb_c.get("count", 0)
            aa, ab = sa_c.get("avg_conf", 0.0), sb_c.get("avg_conf", 0.0)
            better = la if aa > ab else (lb if ab > aa else "Tie")
            print(f"  {cls:<20} {ca:>6}  {aa:>9.4f}  {cb:>6}  {ab:>9.4f}  {better:>10}")
        sep("═")

    print()
    print("  NOTE: All scores are proxy estimates — no ground truth used.")
    print("  ◀  marks the better value per metric row.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Optional bar-chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(cmp: dict, save_path: str = None):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[chart] matplotlib not installed — skipping.")
        return

    la, lb = cmp["labels"]
    ma, mb = cmp[la], cmp[lb]

    entries = [
        ("Avg Confidence",     ma["avg_confidence"],       mb["avg_confidence"],       True),
        ("Temporal Stability", ma["temporal_stability"],   mb["temporal_stability"],   True),
        ("High-Conf Ratio",    ma["high_conf_ratio"],      mb["high_conf_ratio"],      True),
        ("NMS Quality",        ma["nms_quality_score"],    mb["nms_quality_score"],    True),
        ("Det Coverage",       ma["detection_coverage"],   mb["detection_coverage"],   True),
        ("Low-Conf Ratio",     ma["low_conf_ratio"],       mb["low_conf_ratio"],       False),
        ("Conf Std-Dev",       ma["conf_std_dev"],         mb["conf_std_dev"],         False),
        ("Composite /100",     ma["composite_score"]/100,  mb["composite_score"]/100,  True),
    ]

    labels = [e[0] for e in entries]
    va     = [e[1] for e in entries]
    vb     = [e[2] for e in entries]
    hib    = [e[3] for e in entries]

    x, bw = np.arange(len(labels)), 0.35
    fig, ax = plt.subplots(figsize=(14, 6))

    bars_a = ax.bar(x - bw/2, va, bw, label=la, color="#4C72B0", alpha=0.85)
    bars_b = ax.bar(x + bw/2, vb, bw, label=lb, color="#DD8452", alpha=0.85)

    for i, (a, b, h) in enumerate(zip(va, vb, hib)):
        if (a > b) if h else (a < b):
            bars_a[i].set_edgecolor("lime"); bars_a[i].set_linewidth(2.5)
        if (b > a) if h else (b < a):
            bars_b[i].set_edgecolor("lime"); bars_b[i].set_linewidth(2.5)

    for bar in list(bars_a) + list(bars_b):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Score  (0–1 normalised  or  raw)")
    ax.set_title("YOLO Model Comparison — Proxy Accuracy Metrics")
    ax.set_ylim(0, max(max(va), max(vb)) * 1.18)
    ax.legend(handles=[
        mpatches.Patch(color="#4C72B0", label=la),
        mpatches.Patch(color="#DD8452", label=lb),
        mpatches.Patch(edgecolor="lime", facecolor="none",
                       linewidth=2, label="Winner per metric"),
    ])
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[chart] Saved → {save_path}")
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Compare two YOLO models on a test video with annotated output")
    ap.add_argument("video",          help="Test video path")
    ap.add_argument("model_a",        help="First  YOLO model (.pt)")
    ap.add_argument("model_b",        help="Second YOLO model (.pt)")
    ap.add_argument("--label-a",      default="Model A",
                    help="Display name for model A  (default: 'Model A')")
    ap.add_argument("--label-b",      default="Model B",
                    help="Display name for model B  (default: 'Model B')")
    ap.add_argument("--conf",         type=float, default=0.25,
                    help="Confidence threshold  (default: 0.25)")
    ap.add_argument("--iou",          type=float, default=0.4,
                    help="IoU threshold for box matching  (default: 0.4)")
    ap.add_argument("--show",         action="store_true",
                    help="Show live side-by-side window while processing")
    ap.add_argument("--save-video",   metavar="PATH",
                    help="Save annotated comparison video  (e.g. output.mp4)")
    ap.add_argument("--chart",        action="store_true",
                    help="Display metric comparison bar chart")
    ap.add_argument("--save-chart",   metavar="PATH",
                    help="Save chart image  (e.g. chart.png)")
    ap.add_argument("--save-json",    metavar="PATH",
                    help="Save full metrics to JSON  (e.g. results.json)")
    args = ap.parse_args()

    comparator = YOLOModelComparator(
        model_a=args.model_a,
        model_b=args.model_b,
        conf=args.conf,
        iou_thr=args.iou,
        label_a=args.label_a,
        label_b=args.label_b,
    )

    results = comparator.run(
        args.video,
        show=args.show,
        save_video=args.save_video,
    )

    print_report(results)

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[json] Saved → {args.save_json}")

    if args.chart or args.save_chart:
        plot_comparison(results, save_path=args.save_chart)

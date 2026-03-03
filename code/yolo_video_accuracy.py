"""
YOLO Video Inference Accuracy Estimator (No Ground Truth Required)
-------------------------------------------------------------------
Since no ground truth is available, accuracy is estimated using:
  1. Average detection confidence       -> how certain the model is
  2. Temporal stability score           -> consistency across frames
  3. Confidence distribution tightness  -> low std-dev = reliable
  4. Per-class reliability scores       -> per-class confidence stats
  5. NMS overlap quality                -> clean detections = lower score
"""

import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def iou(box1, box2):
    """Intersection-over-Union between two [x1, y1, x2, y2] boxes."""
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def box_center(box):
    """Return (cx, cy) of a [x1, y1, x2, y2] box."""
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def center_distance(b1, b2):
    c1, c2 = box_center(b1), box_center(b2)
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5


# ──────────────────────────────────────────────────────────────────────────────
# Core analyser
# ──────────────────────────────────────────────────────────────────────────────

class YOLOVideoAccuracyEstimator:
    def __init__(self, model_path: str, conf_threshold: float = 0.25,
                 iou_threshold: float = 0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold   = iou_threshold
        self._reset()

    # ── state ─────────────────────────────────────────────────────────────────

    def _reset(self):
        self.all_confidences   = []          # every detection confidence
        self.per_class_conf    = defaultdict(list)
        self.per_frame_counts  = []          # #detections per frame
        self.temporal_stab     = []          # stability score per frame
        self.nms_overlap_scores = []         # intra-frame avg-IoU
        self.prev_boxes        = []          # boxes from previous frame
        self.frame_idx         = 0

    # ── per-frame processing ──────────────────────────────────────────────────

    def _process_frame(self, results):
        """Extract metrics from a single frame result."""
        result = results[0]
        
        # Try OBB (Oriented Bounding Box) first, then fall back to standard boxes
        boxes = result.obb if result.obb is not None else result.boxes
        
        if boxes is None or len(boxes) == 0:
            self.per_frame_counts.append(0)
            if self.prev_boxes:
                self.temporal_stab.append(0.0)   # objects disappeared
            self.prev_boxes = []
            return

        confs   = boxes.conf
        if confs is None:
            self.per_frame_counts.append(0)
            self.prev_boxes = []
            return
        
        confs   = confs.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        xyxy    = boxes.xyxy.cpu().numpy()        # [N, 4]

        # 1. Confidence accumulation
        self.all_confidences.extend(confs.tolist())
        for c, cls in zip(confs, cls_ids):
            name = self.model.names[cls]
            self.per_class_conf[name].append(float(c))

        # 2. Detection count
        self.per_frame_counts.append(len(confs))

        # 3. Temporal stability: match current boxes to previous frame boxes
        if self.prev_boxes:
            stab_scores = []
            for cur in xyxy:
                best_iou = max((iou(cur, prev) for prev in self.prev_boxes),
                               default=0.0)
                stab_scores.append(best_iou)
            self.temporal_stab.append(float(np.mean(stab_scores)))

        # 4. Intra-frame NMS overlap quality (lower avg-IoU = cleaner output)
        if len(xyxy) > 1:
            pairs = []
            for i in range(len(xyxy)):
                for j in range(i + 1, len(xyxy)):
                    pairs.append(iou(xyxy[i], xyxy[j]))
            self.nms_overlap_scores.append(float(np.mean(pairs)))
        else:
            self.nms_overlap_scores.append(0.0)

        self.prev_boxes = xyxy.tolist()
        self.frame_idx += 1
    # ── drawing utilities ────────────────────────────────────────────────────

    def _get_boxes_from_result(self, result):
        """Get boxes from result, agnostic of OBB vs standard boxes."""
        # Try OBB first, then fall back to standard boxes
        if result.obb is not None and len(result.obb) > 0:
            return result.obb, 'obb'
        elif result.boxes is not None and len(result.boxes) > 0:
            return result.boxes, 'standard'
        return None, None

    def _draw_boxes(self, frame, results, line_thickness: int = 2):
        """Draw bounding boxes (OBB or standard) with professional appearance."""
        result = results[0]
        boxes, box_type = self._get_boxes_from_result(result)
        
        if boxes is None:
            return frame
        
        # Color palette for different classes
        colors = {
            'small_vehicle': (0, 255, 255),    # Cyan
            'large_vehicle': (0, 165, 255),    # Orange
            'person': (0, 255, 0),             # Green
            'train': (255, 0, 0),              # Red
            'airplane': (255, 0, 255),         # Magenta
            'boat': (255, 255, 0),             # Blue
            'UAV': (0, 255, 127),              # Spring Green
            'UAP': (255, 127, 0),              # Dark Orange
        }
        
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        
        if box_type == 'obb':
            # Oriented Bounding Box with rotated corners
            if hasattr(boxes, 'xyxyxyxy'):
                corners = boxes.xyxyxyxy.cpu().numpy()
                
                for corner_pts, cls_id, conf in zip(corners, cls_ids, confs):
                    cls_name = self.model.names[int(cls_id)]
                    color = colors.get(cls_name, (255, 255, 255))
                    
                    pts = corner_pts.astype(np.int32)
                    cv2.polylines(frame, [pts], True, color, line_thickness, cv2.LINE_AA)
                    
                    text = f"{cls_name} {conf:.2f}"
                    text_pos = tuple(pts[0].astype(int))
                    cv2.putText(frame, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, color, 1, cv2.LINE_AA)
        else:
            # Standard rectangular bounding boxes
            xyxy = boxes.xyxy.cpu().numpy()
            
            for box_coords, cls_id, conf in zip(xyxy, cls_ids, confs):
                x1, y1, x2, y2 = map(int, box_coords)
                cls_name = self.model.names[int(cls_id)]
                color = colors.get(cls_name, (255, 255, 255))
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_thickness, cv2.LINE_AA)
                
                text = f"{cls_name} {conf:.2f}"
                cv2.putText(frame, text, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, color, 1, cv2.LINE_AA)
        
        return frame

    def _draw_metrics_overlay(self, frame, metrics_so_far: dict):
        """Draw real-time metrics on the frame."""
        h, w = frame.shape[:2]
        
        # Create a more professional overlay with border
        overlay = frame.copy()
        
        # Semi-transparent background panel
        panel_x1, panel_y1 = 12, 12
        panel_x2, panel_y2 = 500, 270
        cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        
        # Border for the panel
        cv2.rectangle(frame, (panel_x1, panel_y1), (panel_x2, panel_y2), 
                     (0, 255, 0), 2, cv2.LINE_AA)
        
        # Text configuration (clean, right-aligned values)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.52
        thickness = 1
        label_color = (220, 220, 220)   # light grey for labels
        value_color = (240, 200, 0)     # warm yellow for values
        sep_color = (70, 70, 70)        # separator line
        line_height = 28
        pad = 14
        x_label = panel_x1 + pad
        x_value = panel_x2 - pad
        y = panel_y1 + 28
        
        # Draw metrics with separators for better readability
        metrics_lines = [
            ("Frame", f"{metrics_so_far['frame_idx']}/{metrics_so_far['total_frames']}"),
            ("Current Det.", f"{metrics_so_far['current_detections']}"),
            ("Total Det.", f"{metrics_so_far['total_detections']}"),
            ("Avg Conf.", f"{metrics_so_far['avg_conf']:.3f}"),
            ("High Conf %", f"{metrics_so_far['high_conf_ratio']*100:.1f}%"),
            ("Temporal Stab.", f"{metrics_so_far['temporal_stab']:.3f}"),
            ("Coverage %", f"{metrics_so_far['coverage']*100:.1f}%"),
        ]
        
        # Draw each metric with a subtle separator
        for i, (label, value) in enumerate(metrics_lines):
            # label (left)
            cv2.putText(frame, f"{label}", (x_label, y), font, font_scale, 
                        label_color, thickness, cv2.LINE_AA)

            # value (right-aligned)
            (vw, vh), _ = cv2.getTextSize(str(value), font, font_scale, thickness + 1)
            cv2.putText(frame, str(value), (x_value - vw, y), font, font_scale, 
                        value_color, thickness + 1, cv2.LINE_AA)

            # separator line under each row
            y_line = y + 8
            cv2.line(frame, (panel_x1 + 6, y_line), (panel_x2 - 6, y_line), sep_color, 1)
            y += line_height

        # Draw avg-confidence horizontal bar below metrics
        avg_conf = metrics_so_far.get('avg_conf', 0.0)
        bar_y = panel_y2 - 36
        bar_x1 = panel_x1 + 18
        bar_x2 = panel_x2 - 18
        bar_w = bar_x2 - bar_x1
        filled_w = int(bar_w * max(0.0, min(1.0, avg_conf)))
        cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + 12), (40, 40, 40), -1)
        cv2.rectangle(frame, (bar_x1, bar_y), (bar_x1 + filled_w, bar_y + 12), (0, 200, 0), -1)
        cv2.putText(frame, f"Avg Conf: {avg_conf:.3f}", (bar_x1, bar_y - 6), font, 0.45, value_color, 1, cv2.LINE_AA)
        
        return frame
    # ── public API ────────────────────────────────────────────────────────────

    def run(self, video_path: str, show: bool = False, output_path: str = None):
        """Run inference on the full video and collect metrics."""
        self._reset()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS)
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Video: {video_path}")
        print(f"  Frames: {total_frames}  |  FPS: {fps:.1f}  |  Size: {width}x{height}\n")

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"  Writing output to: {output_path}\n")

        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            self._process_frame(results)

            # Draw boxes (OBB or standard) professionally
            annotated = frame.copy()
            annotated = self._draw_boxes(annotated, results, line_thickness=2)

            # Build current metrics for overlay
            current_det_count = len(results[0].obb) if results[0].obb is not None else 0
            metrics_so_far = {
                'frame_idx': frame_num + 1,
                'total_frames': total_frames,
                'current_detections': current_det_count,
                'total_detections': len(self.all_confidences),
                'avg_conf': float(np.mean(self.all_confidences)) if self.all_confidences else 0.0,
                'high_conf_ratio': float(np.mean(np.array(self.all_confidences) >= 0.7)) if self.all_confidences else 0.0,
                'temporal_stab': float(np.mean(self.temporal_stab)) if self.temporal_stab else 0.0,
                'coverage': float(np.mean(np.array(self.per_frame_counts) > 0)) if self.per_frame_counts else 0.0,
            }
            
            # Draw metrics on frame
            annotated = self._draw_metrics_overlay(annotated, metrics_so_far)

            if writer:
                writer.write(annotated)

            if show:
                cv2.imshow("YOLO Inference", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_num += 1
            if frame_num % 50 == 0:
                print(f"  Processed {frame_num}/{total_frames} frames …")

        cap.release()
        if writer:
            writer.release()
            print(f"\n  Output saved to: {output_path}")
        if show:
            cv2.destroyAllWindows()

        return self.compute_metrics()

    def compute_metrics(self) -> dict:
        """Aggregate all collected data into final proxy-accuracy metrics."""
        metrics = {}

        # ── 1. Confidence metrics ──────────────────────────────────────────
        if self.all_confidences:
            confs = np.array(self.all_confidences)
            metrics["avg_confidence"]      = float(np.mean(confs))
            metrics["median_confidence"]   = float(np.median(confs))
            metrics["conf_std_dev"]        = float(np.std(confs))
            metrics["high_conf_ratio"]     = float(np.mean(confs >= 0.7))
            metrics["low_conf_ratio"]      = float(np.mean(confs < 0.4))
            metrics["total_detections"]    = len(confs)
        else:
            metrics.update({
                "avg_confidence": 0.0, "median_confidence": 0.0,
                "conf_std_dev": 0.0,   "high_conf_ratio": 0.0,
                "low_conf_ratio": 1.0, "total_detections": 0,
            })

        # ── 2. Temporal stability score (0–1, higher = more stable) ────────
        if self.temporal_stab:
            metrics["temporal_stability"]  = float(np.mean(self.temporal_stab))
        else:
            metrics["temporal_stability"]  = 0.0

        # ── 3. Detection count consistency ─────────────────────────────────
        if self.per_frame_counts:
            counts = np.array(self.per_frame_counts)
            metrics["avg_detections_per_frame"]  = float(np.mean(counts))
            metrics["detection_count_std"]       = float(np.std(counts))
            metrics["frames_with_detections"]    = int(np.sum(counts > 0))
            metrics["total_frames_processed"]    = len(counts)
            metrics["detection_coverage"]        = float(
                np.mean(counts > 0))   # fraction of frames with ≥1 detection
        else:
            metrics.update({
                "avg_detections_per_frame": 0.0,
                "detection_count_std":      0.0,
                "frames_with_detections":   0,
                "total_frames_processed":   0,
                "detection_coverage":       0.0,
            })

        # ── 4. NMS quality (lower avg intra-frame IoU = cleaner boxes) ─────
        if self.nms_overlap_scores:
            metrics["avg_intra_frame_iou"] = float(
                np.mean(self.nms_overlap_scores))
            # Convert to a 0-1 quality score (1 = perfectly clean)
            metrics["nms_quality_score"]   = float(
                1.0 - np.mean(self.nms_overlap_scores))
        else:
            metrics["avg_intra_frame_iou"] = 0.0
            metrics["nms_quality_score"]   = 1.0

        # ── 5. Per-class stats ─────────────────────────────────────────────
        per_class = {}
        for cls, confs in self.per_class_conf.items():
            arr = np.array(confs)
            per_class[cls] = {
                "count":          len(arr),
                "avg_confidence": float(np.mean(arr)),
                "min_confidence": float(np.min(arr)),
                "max_confidence": float(np.max(arr)),
            }
        metrics["per_class_stats"] = per_class

        # ── 6. Composite proxy-accuracy score (0–100) ──────────────────────
        # Weighted combination of the above sub-scores
        w_conf  = 0.40   # avg confidence
        w_stab  = 0.30   # temporal stability
        w_nms   = 0.15   # NMS quality
        w_hconf = 0.15   # ratio of high-confidence detections

        composite = (
            w_conf  * metrics["avg_confidence"]
          + w_stab  * metrics["temporal_stability"]
          + w_nms   * metrics["nms_quality_score"]
          + w_hconf * metrics["high_conf_ratio"]
        ) * 100

        metrics["composite_proxy_accuracy"] = round(composite, 2)

        return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Report printer
# ──────────────────────────────────────────────────────────────────────────────

def print_report(metrics: dict):
    sep = "─" * 60
    print(f"\n{sep}")
    print("  YOLO VIDEO INFERENCE ACCURACY REPORT (no ground truth)")
    print(sep)

    print("\n[Confidence Metrics]")
    print(f"  Average confidence      : {metrics['avg_confidence']:.4f}")
    print(f"  Median  confidence      : {metrics['median_confidence']:.4f}")
    print(f"  Confidence std-dev      : {metrics['conf_std_dev']:.4f}")
    print(f"  High-conf ratio (≥0.70) : {metrics['high_conf_ratio']*100:.1f}%")
    print(f"  Low-conf  ratio (<0.40) : {metrics['low_conf_ratio']*100:.1f}%")
    print(f"  Total detections        : {metrics['total_detections']}")

    print("\n[Temporal Stability]")
    print(f"  Stability score (0–1)   : {metrics['temporal_stability']:.4f}")
    print("  (higher = detections are consistent frame-to-frame)")

    print("\n[Detection Coverage]")
    print(f"  Frames processed        : {metrics['total_frames_processed']}")
    print(f"  Frames with detections  : {metrics['frames_with_detections']}")
    print(f"  Detection coverage      : {metrics['detection_coverage']*100:.1f}%")
    print(f"  Avg detections / frame  : {metrics['avg_detections_per_frame']:.2f}")
    print(f"  Detection count std-dev : {metrics['detection_count_std']:.2f}")

    print("\n[NMS / Box Quality]")
    print(f"  Avg intra-frame IoU     : {metrics['avg_intra_frame_iou']:.4f}")
    print(f"  NMS quality score (0–1) : {metrics['nms_quality_score']:.4f}")
    print("  (closer to 1 = minimal box overlap = cleaner detections)")

    print("\n[Per-Class Statistics]")
    if metrics["per_class_stats"]:
        header = f"  {'Class':<20} {'Count':>6}  {'Avg Conf':>9}  {'Min':>6}  {'Max':>6}"
        print(header)
        print("  " + "-" * 56)
        for cls, s in sorted(metrics["per_class_stats"].items(),
                              key=lambda x: -x[1]["count"]):
            print(f"  {cls:<20} {s['count']:>6}  {s['avg_confidence']:>9.4f}"
                  f"  {s['min_confidence']:>6.4f}  {s['max_confidence']:>6.4f}")
    else:
        print("  No detections recorded.")

    print(f"\n{'─'*60}")
    score = metrics["composite_proxy_accuracy"]
    grade = ("Excellent" if score >= 80 else
             "Good"      if score >= 65 else
             "Fair"      if score >= 50 else
             "Poor")
    print(f"  COMPOSITE PROXY-ACCURACY SCORE : {score:.2f} / 100  [{grade}]")
    print(f"{'─'*60}\n")
    print("NOTE: This score is an ESTIMATE based on model confidence and")
    print("temporal consistency — not a ground-truth accuracy measurement.\n")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Estimate YOLO model accuracy on a video (no ground truth)")
    parser.add_argument("video",        help="Path to input video file")
    parser.add_argument("--model",      default="yolov8n.pt",
                        help="YOLO model path (default: yolov8n.pt)")
    parser.add_argument("--conf",       type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou",        type=float, default=0.5,
                        help="IoU threshold for NMS (default: 0.5)")
    parser.add_argument("--show",       action="store_true",
                        help="Display annotated video while processing")
    parser.add_argument("--output",     default=None,
                        help="Path to save the annotated output video (e.g. out.mp4)")
    args = parser.parse_args()

    estimator = YOLOVideoAccuracyEstimator(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )

    metrics = estimator.run(args.video, show=args.show, output_path=args.output)
    print_report(metrics)

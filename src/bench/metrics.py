"""
Evaluation Metrics for Floor Plan Analysis.

Room Segmentation Metrics:
- IoU (Intersection over Union) at 0.5 and 0.75 thresholds
- Precision / Recall / F1
- Area error (% difference) per matched room
- Label accuracy (exact + alias match)

Opening Detection Metrics:
- Bbox IoU
- Precision / Recall by type (door/window/ventilator)

Scale Metrics:
- Scale error %
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

try:
    from shapely.geometry import Polygon
    from shapely.validation import make_valid
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class RoomMatch:
    """Match between predicted and ground truth room."""
    pred_id: str
    gt_id: str
    iou: float
    pred_label: str
    gt_label: str
    label_match: bool
    pred_area: float
    gt_area: float
    area_error_pct: float


@dataclass
class RoomMetrics:
    """Room segmentation metrics."""
    # IoU thresholds
    precision_50: float = 0.0
    recall_50: float = 0.0
    f1_50: float = 0.0

    precision_75: float = 0.0
    recall_75: float = 0.0
    f1_75: float = 0.0

    # Label accuracy
    label_accuracy: float = 0.0
    label_accuracy_with_alias: float = 0.0

    # Area metrics
    mean_area_error_pct: float = 0.0
    max_area_error_pct: float = 0.0

    # Counts
    num_pred: int = 0
    num_gt: int = 0
    num_matched_50: int = 0
    num_matched_75: int = 0

    # Segmentation issues
    oversegmented: int = 0  # GT matched by multiple preds
    undersegmented: int = 0  # Multiple GTs matched by single pred
    missed: int = 0  # GT not matched
    spurious: int = 0  # Pred not matched

    # Detailed matches
    matches: List[RoomMatch] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "precision_50": self.precision_50,
            "recall_50": self.recall_50,
            "f1_50": self.f1_50,
            "precision_75": self.precision_75,
            "recall_75": self.recall_75,
            "f1_75": self.f1_75,
            "label_accuracy": self.label_accuracy,
            "label_accuracy_with_alias": self.label_accuracy_with_alias,
            "mean_area_error_pct": self.mean_area_error_pct,
            "max_area_error_pct": self.max_area_error_pct,
            "num_pred": self.num_pred,
            "num_gt": self.num_gt,
            "num_matched_50": self.num_matched_50,
            "num_matched_75": self.num_matched_75,
            "oversegmented": self.oversegmented,
            "undersegmented": self.undersegmented,
            "missed": self.missed,
            "spurious": self.spurious,
        }


@dataclass
class OpeningMatch:
    """Match between predicted and ground truth opening."""
    pred_id: str
    gt_id: str
    iou: float
    pred_type: str
    gt_type: str
    type_match: bool


@dataclass
class OpeningMetrics:
    """Opening detection metrics."""
    # Overall
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    # By type
    door_precision: float = 0.0
    door_recall: float = 0.0
    door_f1: float = 0.0

    window_precision: float = 0.0
    window_recall: float = 0.0
    window_f1: float = 0.0

    ventilator_precision: float = 0.0
    ventilator_recall: float = 0.0
    ventilator_f1: float = 0.0

    # Counts
    num_pred: int = 0
    num_gt: int = 0
    num_matched: int = 0

    # Detailed matches
    matches: List[OpeningMatch] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "door_precision": self.door_precision,
            "door_recall": self.door_recall,
            "door_f1": self.door_f1,
            "window_precision": self.window_precision,
            "window_recall": self.window_recall,
            "window_f1": self.window_f1,
            "ventilator_precision": self.ventilator_precision,
            "ventilator_recall": self.ventilator_recall,
            "ventilator_f1": self.ventilator_f1,
            "num_pred": self.num_pred,
            "num_gt": self.num_gt,
            "num_matched": self.num_matched,
        }


@dataclass
class ScaleMetrics:
    """Scale detection metrics."""
    has_gt_scale: bool = False
    has_pred_scale: bool = False
    scale_error_pct: float = 0.0
    gt_px_per_mm: float = 0.0
    pred_px_per_mm: float = 0.0
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_gt_scale": self.has_gt_scale,
            "has_pred_scale": self.has_pred_scale,
            "scale_error_pct": self.scale_error_pct,
            "gt_px_per_mm": self.gt_px_per_mm,
            "pred_px_per_mm": self.pred_px_per_mm,
            "confidence": self.confidence,
        }


def compute_polygon_iou(
    poly1: List[Tuple[float, float]],
    poly2: List[Tuple[float, float]]
) -> float:
    """Compute IoU between two polygons."""
    if not SHAPELY_AVAILABLE:
        # Fallback: bbox IoU
        return compute_bbox_iou_from_polygons(poly1, poly2)

    try:
        p1 = Polygon(poly1)
        p2 = Polygon(poly2)

        if not p1.is_valid:
            p1 = make_valid(p1)
        if not p2.is_valid:
            p2 = make_valid(p2)

        if p1.area == 0 or p2.area == 0:
            return 0.0

        intersection = p1.intersection(p2).area
        union = p1.union(p2).area

        if union == 0:
            return 0.0

        return intersection / union

    except Exception as e:
        logger.warning(f"IoU computation failed: {e}")
        return 0.0


def compute_bbox_iou(
    bbox1: Tuple[int, int, int, int],
    bbox2: Tuple[int, int, int, int]
) -> float:
    """Compute IoU between two bounding boxes (x, y, w, h)."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Convert to corners
    x1_max, y1_max = x1 + w1, y1 + h1
    x2_max, y2_max = x2 + w2, y2 + h2

    # Intersection
    xi = max(x1, x2)
    yi = max(y1, y2)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)

    if xi_max <= xi or yi_max <= yi:
        return 0.0

    intersection = (xi_max - xi) * (yi_max - yi)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def compute_bbox_iou_from_polygons(
    poly1: List[Tuple[float, float]],
    poly2: List[Tuple[float, float]]
) -> float:
    """Compute bbox IoU as fallback when shapely unavailable."""
    def poly_to_bbox(poly):
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        return (int(min(xs)), int(min(ys)), int(max(xs) - min(xs)), int(max(ys) - min(ys)))

    return compute_bbox_iou(poly_to_bbox(poly1), poly_to_bbox(poly2))


def compute_room_metrics(
    pred_rooms: List[Dict[str, Any]],
    gt_rooms: List[Dict[str, Any]],
    iou_threshold_50: float = 0.5,
    iou_threshold_75: float = 0.75,
) -> RoomMetrics:
    """
    Compute room segmentation metrics.

    Args:
        pred_rooms: List of predicted rooms with 'polygon', 'label', 'area' keys
        gt_rooms: List of GT rooms with 'polygon', 'label', 'area' keys
        iou_threshold_50: IoU threshold for @0.5 metrics
        iou_threshold_75: IoU threshold for @0.75 metrics

    Returns:
        RoomMetrics
    """
    from .annotation import labels_match

    metrics = RoomMetrics(
        num_pred=len(pred_rooms),
        num_gt=len(gt_rooms),
    )

    if not pred_rooms or not gt_rooms:
        return metrics

    # Compute IoU matrix
    iou_matrix = np.zeros((len(pred_rooms), len(gt_rooms)))
    for i, pred in enumerate(pred_rooms):
        for j, gt in enumerate(gt_rooms):
            iou_matrix[i, j] = compute_polygon_iou(
                pred.get("polygon", []),
                gt.get("polygon", [])
            )

    # Greedy matching at IoU 0.5
    matched_pred_50 = set()
    matched_gt_50 = set()
    matches_50 = []

    # Sort by IoU descending
    indices = np.unravel_index(np.argsort(-iou_matrix, axis=None), iou_matrix.shape)
    for pi, gi in zip(indices[0], indices[1]):
        if pi in matched_pred_50 or gi in matched_gt_50:
            continue
        iou = iou_matrix[pi, gi]
        if iou >= iou_threshold_50:
            matched_pred_50.add(pi)
            matched_gt_50.add(gi)
            matches_50.append((pi, gi, iou))

    # Matching at IoU 0.75
    matched_pred_75 = set()
    matched_gt_75 = set()
    for pi, gi, iou in matches_50:
        if iou >= iou_threshold_75:
            matched_pred_75.add(pi)
            matched_gt_75.add(gi)

    # Compute precision/recall/F1 at 0.5
    tp_50 = len(matches_50)
    metrics.num_matched_50 = tp_50
    metrics.precision_50 = tp_50 / len(pred_rooms) if pred_rooms else 0
    metrics.recall_50 = tp_50 / len(gt_rooms) if gt_rooms else 0
    metrics.f1_50 = 2 * metrics.precision_50 * metrics.recall_50 / (metrics.precision_50 + metrics.recall_50 + 1e-9)

    # At 0.75
    tp_75 = len(matched_pred_75)
    metrics.num_matched_75 = tp_75
    metrics.precision_75 = tp_75 / len(pred_rooms) if pred_rooms else 0
    metrics.recall_75 = tp_75 / len(gt_rooms) if gt_rooms else 0
    metrics.f1_75 = 2 * metrics.precision_75 * metrics.recall_75 / (metrics.precision_75 + metrics.recall_75 + 1e-9)

    # Label accuracy and area errors
    label_matches_exact = 0
    label_matches_alias = 0
    area_errors = []

    for pi, gi, iou in matches_50:
        pred = pred_rooms[pi]
        gt = gt_rooms[gi]

        pred_label = pred.get("label", "Room")
        gt_label = gt.get("label", "Room")

        # Remove numbers from labels for comparison (e.g., "Bedroom 1" -> "Bedroom")
        pred_label_base = ' '.join(pred_label.split()[:-1]) if pred_label.split()[-1].isdigit() else pred_label
        gt_label_base = ' '.join(gt_label.split()[:-1]) if gt_label.split()[-1].isdigit() else gt_label

        exact_match = pred_label_base.lower() == gt_label_base.lower()
        alias_match = labels_match(pred_label_base, gt_label_base)

        if exact_match:
            label_matches_exact += 1
        if alias_match:
            label_matches_alias += 1

        # Area error
        pred_area = pred.get("area", 0)
        gt_area = gt.get("area", 0)
        if gt_area > 0:
            area_error = abs(pred_area - gt_area) / gt_area * 100
            area_errors.append(area_error)
        else:
            area_error = 0

        # Store match details
        metrics.matches.append(RoomMatch(
            pred_id=pred.get("id", str(pi)),
            gt_id=gt.get("id", str(gi)),
            iou=iou,
            pred_label=pred_label,
            gt_label=gt_label,
            label_match=alias_match,
            pred_area=pred_area,
            gt_area=gt_area,
            area_error_pct=area_error,
        ))

    metrics.label_accuracy = label_matches_exact / len(matches_50) if matches_50 else 0
    metrics.label_accuracy_with_alias = label_matches_alias / len(matches_50) if matches_50 else 0
    metrics.mean_area_error_pct = np.mean(area_errors) if area_errors else 0
    metrics.max_area_error_pct = max(area_errors) if area_errors else 0

    # Segmentation issues
    metrics.missed = len(gt_rooms) - len(matched_gt_50)
    metrics.spurious = len(pred_rooms) - len(matched_pred_50)

    # Detect over/under segmentation
    # Over-segmentation: one GT matched by multiple preds
    gt_match_counts = {}
    for pi, gi, _ in matches_50:
        gt_match_counts[gi] = gt_match_counts.get(gi, 0) + 1
    metrics.oversegmented = sum(1 for c in gt_match_counts.values() if c > 1)

    # Under-segmentation: check if pred matches multiple GT significantly
    # This is harder to detect, use IoU overlap heuristic
    for pi in range(len(pred_rooms)):
        high_overlap_count = sum(1 for gi in range(len(gt_rooms)) if iou_matrix[pi, gi] > 0.3)
        if high_overlap_count > 1:
            metrics.undersegmented += 1

    return metrics


def compute_opening_metrics(
    pred_openings: List[Dict[str, Any]],
    gt_openings: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
) -> OpeningMetrics:
    """
    Compute opening detection metrics.

    Args:
        pred_openings: List of predicted openings with 'bbox', 'type' keys
        gt_openings: List of GT openings with 'bbox', 'type' keys
        iou_threshold: IoU threshold for matching

    Returns:
        OpeningMetrics
    """
    metrics = OpeningMetrics(
        num_pred=len(pred_openings),
        num_gt=len(gt_openings),
    )

    if not pred_openings or not gt_openings:
        return metrics

    # Compute IoU matrix
    iou_matrix = np.zeros((len(pred_openings), len(gt_openings)))
    for i, pred in enumerate(pred_openings):
        for j, gt in enumerate(gt_openings):
            iou_matrix[i, j] = compute_bbox_iou(
                tuple(pred.get("bbox", (0, 0, 0, 0))),
                tuple(gt.get("bbox", (0, 0, 0, 0)))
            )

    # Greedy matching
    matched_pred = set()
    matched_gt = set()
    matches = []

    indices = np.unravel_index(np.argsort(-iou_matrix, axis=None), iou_matrix.shape)
    for pi, gi in zip(indices[0], indices[1]):
        if pi in matched_pred or gi in matched_gt:
            continue
        iou = iou_matrix[pi, gi]
        if iou >= iou_threshold:
            matched_pred.add(pi)
            matched_gt.add(gi)

            pred = pred_openings[pi]
            gt = gt_openings[gi]

            matches.append(OpeningMatch(
                pred_id=pred.get("id", str(pi)),
                gt_id=gt.get("id", str(gi)),
                iou=iou,
                pred_type=pred.get("type", "unknown"),
                gt_type=gt.get("type", "unknown"),
                type_match=pred.get("type", "").lower() == gt.get("type", "").lower(),
            ))

    metrics.num_matched = len(matches)
    metrics.matches = matches

    # Overall metrics
    tp = len(matches)
    metrics.precision = tp / len(pred_openings) if pred_openings else 0
    metrics.recall = tp / len(gt_openings) if gt_openings else 0
    metrics.f1 = 2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall + 1e-9)

    # Per-type metrics
    for opening_type in ["door", "window", "ventilator"]:
        gt_type = [g for g in gt_openings if g.get("type", "").lower() == opening_type]
        pred_type = [p for p in pred_openings if p.get("type", "").lower() == opening_type]
        matched_type = [m for m in matches if m.gt_type.lower() == opening_type and m.type_match]

        tp_type = len(matched_type)
        prec = tp_type / len(pred_type) if pred_type else 0
        rec = tp_type / len(gt_type) if gt_type else 0
        f1 = 2 * prec * rec / (prec + rec + 1e-9)

        if opening_type == "door":
            metrics.door_precision = prec
            metrics.door_recall = rec
            metrics.door_f1 = f1
        elif opening_type == "window":
            metrics.window_precision = prec
            metrics.window_recall = rec
            metrics.window_f1 = f1
        elif opening_type == "ventilator":
            metrics.ventilator_precision = prec
            metrics.ventilator_recall = rec
            metrics.ventilator_f1 = f1

    return metrics


def compute_scale_metrics(
    pred_scale: Optional[Dict[str, Any]],
    gt_scale: Optional[Dict[str, Any]],
) -> ScaleMetrics:
    """
    Compute scale detection metrics.

    Args:
        pred_scale: Predicted scale with 'px_per_mm', 'confidence'
        gt_scale: GT scale with 'point1', 'point2', 'length_mm'

    Returns:
        ScaleMetrics
    """
    metrics = ScaleMetrics(
        has_gt_scale=gt_scale is not None,
        has_pred_scale=pred_scale is not None,
    )

    if not gt_scale:
        return metrics

    # Compute GT px_per_mm
    if gt_scale.get("point1") and gt_scale.get("point2") and gt_scale.get("length_mm"):
        p1 = gt_scale["point1"]
        p2 = gt_scale["point2"]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        px_dist = np.sqrt(dx**2 + dy**2)
        metrics.gt_px_per_mm = px_dist / gt_scale["length_mm"] if gt_scale["length_mm"] > 0 else 0

    if not pred_scale:
        metrics.scale_error_pct = 100.0  # No prediction = 100% error
        return metrics

    metrics.pred_px_per_mm = pred_scale.get("px_per_mm", 0)
    metrics.confidence = pred_scale.get("confidence", 0)

    if metrics.gt_px_per_mm > 0:
        error = abs(metrics.pred_px_per_mm - metrics.gt_px_per_mm) / metrics.gt_px_per_mm * 100
        metrics.scale_error_pct = error

    return metrics


if __name__ == "__main__":
    # Test metrics computation
    pred_rooms = [
        {"id": "R1", "polygon": [(0, 0), (100, 0), (100, 100), (0, 100)], "label": "Bedroom", "area": 10000},
        {"id": "R2", "polygon": [(110, 0), (200, 0), (200, 80), (110, 80)], "label": "Kitchen", "area": 7200},
    ]

    gt_rooms = [
        {"id": "GT1", "polygon": [(5, 5), (95, 5), (95, 95), (5, 95)], "label": "Bed Room", "area": 8100},
        {"id": "GT2", "polygon": [(105, 5), (195, 5), (195, 85), (105, 85)], "label": "Kitchen", "area": 7200},
    ]

    metrics = compute_room_metrics(pred_rooms, gt_rooms)
    print("Room Metrics:")
    print(f"  Precision@0.5: {metrics.precision_50:.2%}")
    print(f"  Recall@0.5: {metrics.recall_50:.2%}")
    print(f"  F1@0.5: {metrics.f1_50:.2%}")
    print(f"  Label Accuracy (alias): {metrics.label_accuracy_with_alias:.2%}")
    print(f"  Mean Area Error: {metrics.mean_area_error_pct:.1f}%")

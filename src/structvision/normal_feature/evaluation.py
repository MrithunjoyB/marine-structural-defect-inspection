"""Dense development-only PatchCore metrics; no probability interpretation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class DenseDevelopmentMetrics:
    image_count: int
    positive_image_count: int
    clean_image_count: int
    pixel_average_precision: float
    image_average_precision: float
    image_roc_auc: float
    au_pro_fpr_limit_0_30: float
    au_pro_threshold_count: int
    anomaly_map_distributions: tuple[tuple[str, tuple[tuple[str, float], ...]], ...]
    score_semantics: str = "raw_patchcore_distance_not_probability"
    evidence_classification: str = "development-only — non-confirmatory"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _distribution(values: np.ndarray) -> tuple[tuple[str, float], ...]:
    return (
        ("mean", float(np.mean(values))),
        ("std", float(np.std(values))),
        ("p50", float(np.quantile(values, 0.50))),
        ("p95", float(np.quantile(values, 0.95))),
        ("p99", float(np.quantile(values, 0.99))),
        ("maximum", float(np.max(values))),
    )


def dense_development_metrics(
    *,
    anomaly_maps: tuple[np.ndarray, ...],
    image_scores: tuple[float, ...],
    ground_truths: tuple[np.ndarray, ...],
    outcomes: tuple[str, ...],
    categories: tuple[str, ...],
    au_pro_threshold_count: int = 101,
) -> DenseDevelopmentMetrics:
    if not anomaly_maps or not (
        len(anomaly_maps) == len(image_scores) == len(ground_truths) == len(outcomes) == len(categories)
    ):
        raise ValueError("Dense metric inputs must be non-empty and aligned")
    maps = tuple(np.ascontiguousarray(item, dtype=np.float32) for item in anomaly_maps)
    truths = tuple(np.ascontiguousarray(item > 0, dtype=np.uint8) for item in ground_truths)
    if any(left.shape != right.shape for left, right in zip(maps, truths)):
        raise ValueError("Dense maps and ground truths must be pixel-aligned")
    if set(outcomes) != {"no_anomaly", "anomaly_present"}:
        raise ValueError("Dense development metrics require clean and positive validation images")
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ImportError as error:
        raise RuntimeError("Dense learned metrics require the exact optional scikit-learn dependency") from error
    pixel_truth = np.concatenate([item.reshape(-1) for item in truths])
    pixel_scores = np.concatenate([item.reshape(-1) for item in maps])
    labels = np.asarray([item == "anomaly_present" for item in outcomes], dtype=np.uint8)
    scores = np.asarray(image_scores, dtype=np.float64)
    thresholds = np.unique(np.quantile(
        pixel_scores, np.linspace(0.0, 1.0, au_pro_threshold_count), method="linear",
    ))[::-1]
    thresholds = np.concatenate((
        [np.nextafter(np.float32(pixel_scores.max()), np.float32(np.inf))], thresholds,
    ))
    clean_indices = [index for index, outcome in enumerate(outcomes) if outcome == "no_anomaly"]
    positive_indices = [index for index, outcome in enumerate(outcomes) if outcome == "anomaly_present"]
    clean_pixels = sum(maps[index].size for index in clean_indices)
    region_masks = []
    for index in positive_indices:
        count, components = cv2.connectedComponents(truths[index], connectivity=8)
        region_masks.extend((index, components == label) for label in range(1, count))
    curve = []
    for threshold in thresholds:
        false_pixels = sum(int(np.count_nonzero(maps[index] >= threshold)) for index in clean_indices)
        fpr = false_pixels / clean_pixels
        overlaps = [
            np.count_nonzero((maps[index] >= threshold) & region) / np.count_nonzero(region)
            for index, region in region_masks
        ]
        curve.append((fpr, float(np.mean(overlaps)) if overlaps else 0.0))
    by_fpr: dict[float, float] = {}
    for fpr, pro in curve:
        if fpr <= 0.30 + 1e-12:
            by_fpr[fpr] = max(pro, by_fpr.get(fpr, 0.0))
    points = sorted(by_fpr.items())
    if not points:
        au_pro = 0.0
    else:
        if points[0][0] > 0:
            points.insert(0, (0.0, 0.0))
        if points[-1][0] < 0.30:
            points.append((0.30, points[-1][1]))
        au_pro = float(np.trapezoid(
            np.asarray([item[1] for item in points]),
            np.asarray([item[0] for item in points]),
        ) / 0.30)
    distribution_groups: dict[str, list[np.ndarray]] = {}
    for anomaly_map, outcome, category in zip(maps, outcomes, categories):
        distribution_groups.setdefault(f"outcome:{outcome}", []).append(anomaly_map.reshape(-1))
        distribution_groups.setdefault(f"category:{category}", []).append(anomaly_map.reshape(-1))
    distributions = tuple(
        (name, _distribution(np.concatenate(values)))
        for name, values in sorted(distribution_groups.items())
    )
    return DenseDevelopmentMetrics(
        image_count=len(maps),
        positive_image_count=len(positive_indices),
        clean_image_count=len(clean_indices),
        pixel_average_precision=float(average_precision_score(pixel_truth, pixel_scores)),
        image_average_precision=float(average_precision_score(labels, scores)),
        image_roc_auc=float(roc_auc_score(labels, scores)),
        au_pro_fpr_limit_0_30=au_pro,
        au_pro_threshold_count=len(thresholds),
        anomaly_map_distributions=distributions,
    )

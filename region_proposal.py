"""Multi-scale, explainable classical-CV anomaly region proposals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from config import MASK_DIR, OUTPUT_DIR
from feature_extraction import FeatureMaps, extract_feature_maps
from scoring import PriorityResult, priority_label, score_architecture


@dataclass(frozen=True)
class AblationConfig:
    edge_features: bool = True
    texture_features: bool = True
    colour_features: bool = True
    entropy_features: bool = True
    stability: bool = True
    contextual_contrast: bool = True
    multi_scale_fusion: bool = True
    region_merging: bool = True
    mask_refinement: bool = True


@dataclass(frozen=True)
class ProposalDiagnostics:
    raw_components: int
    after_filtering: int
    after_merging: int
    heatmap_threshold: float
    score_distribution: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        scores = np.asarray(self.score_distribution, dtype=float)
        return {
            "Raw connected components": self.raw_components,
            "After filtering": self.after_filtering,
            "After merging": self.after_merging,
            "Heatmap threshold": round(self.heatmap_threshold, 1),
            "Minimum score": round(float(scores.min()), 1) if scores.size else 0.0,
            "Median score": round(float(np.median(scores)), 1) if scores.size else 0.0,
            "Maximum score": round(float(scores.max()), 1) if scores.size else 0.0,
        }


@dataclass(frozen=True)
class RegionProposal:
    region_id: str
    bbox: tuple[int, int, int, int]
    pixel_area: int
    relative_area: float
    aspect_ratio: float
    perimeter: float
    edge_density: float
    texture_score: float
    color_variation_score: float
    gradient_strength: float
    entropy: float
    contrast_difference: float
    mask_stability: float
    local_texture_contrast: float
    local_colour_contrast: float
    local_entropy_contrast: float
    internal_vs_boundary_edge_ratio: float
    anomaly_evidence_score: float
    mask_reliability_score: float
    coherence_score: float
    border_penalty: float
    area_reduction: float
    boundary_smoothness: float
    feature_contributions: dict[str, float]
    dominant_features: tuple[str, ...]
    explanation: str
    priority: PriorityResult
    mask_path: Path
    raw_mask_path: Path
    context_mask_path: Path

    def to_row(self) -> dict[str, object]:
        x1, y1, x2, y2 = self.bbox
        return {
            "Region ID": self.region_id, "BBox": f"({x1}, {y1}) - ({x2}, {y2})",
            "Pixel Area": self.pixel_area, "Relative Area (%)": round(self.relative_area * 100, 3),
            "Aspect Ratio": round(self.aspect_ratio, 2), "Perimeter": round(self.perimeter, 1),
            "Edge Density": round(self.edge_density, 3), "Texture Score": round(self.texture_score, 3),
            "Color Variation": round(self.color_variation_score, 3), "Gradient": round(self.gradient_strength, 3),
            "Entropy": round(self.entropy, 3), "Stability": round(self.mask_stability, 3),
            "Local Texture Contrast": round(self.local_texture_contrast, 3),
            "Local Colour Contrast": round(self.local_colour_contrast, 3),
            "Local Entropy Contrast": round(self.local_entropy_contrast, 3),
            "Internal/Boundary Edge": round(self.internal_vs_boundary_edge_ratio, 3),
            "Anomaly Evidence": round(self.anomaly_evidence_score, 1),
            "Mask Reliability": round(self.mask_reliability_score, 1),
            "Coherence": round(self.coherence_score, 3), "Border Penalty": round(self.border_penalty, 3),
            "Area Reduction (%)": round(self.area_reduction * 100, 1),
            "Boundary Smoothness": round(self.boundary_smoothness, 3),
            "Priority Score": self.priority.score, "Priority Label": self.priority.label,
            "Dominant Features": " + ".join(self.dominant_features), "Why Selected": self.explanation,
        }


@dataclass(frozen=True)
class ProposalResult:
    proposals: list[RegionProposal]
    overlay_path: Path
    combined_mask_path: Path
    diagnostics: ProposalDiagnostics
    comparison_paths: dict[str, Path]
    comparison_counts: dict[str, int]


@dataclass
class _Candidate:
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    raw_mask: np.ndarray | None = None


def propose_regions(
    image: np.ndarray, feature_maps: FeatureMaps, image_stem: str, min_area: int = 250,
    max_regions: int = 20, min_relative_area: float = 0.0002, max_relative_area: float = 0.85,
    score_weights: dict[str, float] | None = None, border_margin: float = 0.025,
    ablation: AblationConfig | None = None,
) -> ProposalResult:
    """Fuse independent feature masks, overlapping tiles, and multi-scale components."""
    height, width = image.shape[:2]
    image_area = max(height * width, 1)
    min_pixels = max(16, min(min_area, int(image_area * 0.01)), int(image_area * min_relative_area))
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ablation = ablation or AblationConfig()
    valid_area = _valid_image_area(image, border_margin)
    fused, threshold, raw_count = _fused_multiscale_mask(image, feature_maps, valid_area, ablation)
    components = _components(fused)
    filtered = [c for c in components if _keep_candidate(c, min_pixels, image_area, max_relative_area, width, height, feature_maps)]
    filtered_count = len(filtered)
    merged = _merge_candidates(filtered, feature_maps, width, height) if ablation.region_merging else filtered
    split = []
    for candidate in merged:
        split.extend(_split_candidate(candidate, feature_maps.anomaly_strength, image_area))
    candidates = []
    for item in split:
        refined=_refine_candidate(item,feature_maps,min_pixels) if ablation.mask_refinement else item
        for coherent_part in _split_candidate(refined,feature_maps.anomaly_strength,image_area):
            candidates.extend(_split_fragmented(coherent_part,min_pixels,image_area))

    raw_metrics: list[tuple[_Candidate, dict[str, float]]] = []
    for candidate in candidates:
        metrics = _region_metrics(image, feature_maps, candidate.mask, candidate.bbox)
        metrics["mask_stability"] = _mask_stability(image, candidate.mask, candidate.bbox) if ablation.stability else 0.5
        metrics.update(_coherence_metrics(candidate.mask, feature_maps.anomaly_strength))
        if candidate.raw_mask is not None:
            raw_area=max(cv2.countNonZero(candidate.raw_mask),1)
            metrics["area_reduction"]=max(0.0,1.0-cv2.countNonZero(candidate.mask)/raw_area)
            metrics["scale_agreement"]=_binary_iou(candidate.mask>0,candidate.raw_mask>0)
        metrics["border_penalty"] = _border_penalty(candidate.mask, valid_area)
        bw=candidate.bbox[2]-candidate.bbox[0]; bh=candidate.bbox[3]-candidate.bbox[1]
        if metrics["border_penalty"]>.75 and max(bw/max(bh,1),bh/max(bw,1))>5 and min(bw,bh)<min(height,width)*.08:
            continue
        raw_metrics.append((candidate, metrics))
    calibrated = _robust_calibrate([metrics for _, metrics in raw_metrics])
    scored: list[tuple[float, _Candidate, dict[str, float], float, float, dict[str, float]]] = []
    for (candidate, metrics), normalized in zip(raw_metrics, calibrated):
        evidence = {
            "local_texture_contrast": normalized["local_texture_contrast"] if ablation.contextual_contrast and ablation.texture_features else 0,
            "local_colour_contrast": normalized["local_colour_contrast"] if ablation.contextual_contrast and ablation.colour_features else 0,
            "local_entropy_contrast": normalized["local_entropy_contrast"] if ablation.contextual_contrast and ablation.entropy_features else 0,
            "edge_concentration": normalized["internal_vs_boundary_edge_ratio"] if ablation.edge_features else 0,
            "gradient_contrast": normalized["gradient_contrast"] if ablation.edge_features else 0,
            "geometric_irregularity": normalized["geometric_irregularity"],
        }
        reliability = {"perturbation_stability": metrics["mask_stability"], "connectedness": metrics["connectedness"],
                       "boundary_smoothness": metrics["boundary_smoothness"], "scale_agreement": metrics["scale_agreement"],
                       "segmentation_coherence": metrics["coherence_score"]}
        evidence_score, reliability_score, score, contributions = score_architecture(
            evidence, reliability, metrics["area_relevance"], normalized["novelty"]
        )
        score *= 1.0 - 0.55 * metrics["border_penalty"]
        scored.append((score, candidate, metrics, evidence_score, reliability_score, contributions))
    scored.sort(key=lambda item: item[0], reverse=True)
    scored = scored[:max_regions]

    proposals: list[RegionProposal] = []
    combined_mask = np.zeros((height, width), np.uint8)
    for index, (score, candidate, metrics, evidence_score, reliability_score, contributions) in enumerate(scored, 1):
        region_id = f"R{index:03d}"
        mask_path = MASK_DIR / f"{image_stem}_{region_id}_mask.png"
        raw_mask_path = MASK_DIR / f"{image_stem}_{region_id}_raw_mask.png"
        context_mask_path = MASK_DIR / f"{image_stem}_{region_id}_context.png"
        cv2.imwrite(str(mask_path), candidate.mask)
        cv2.imwrite(str(raw_mask_path), candidate.raw_mask if candidate.raw_mask is not None else candidate.mask)
        context_mask = _context_ring(candidate.mask)
        cv2.imwrite(str(context_mask_path), context_mask)
        combined_mask = cv2.bitwise_or(combined_mask, candidate.mask)
        x1, y1, x2, y2 = candidate.bbox
        contour_list, _ = cv2.findContours(candidate.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = int(cv2.countNonZero(candidate.mask))
        perimeter = sum(cv2.arcLength(c, True) for c in contour_list)
        aspect = max((x2 - x1) / max(y2 - y1, 1), (y2 - y1) / max(x2 - x1, 1))
        dominant = _dominant_features(contributions)
        explanation = _explain(metrics, aspect, metrics["mask_stability"], area / image_area)
        priority = PriorityResult(round(score, 1), priority_label(score), explanation)
        proposals.append(RegionProposal(
            region_id, candidate.bbox, area, area / image_area, aspect, float(perimeter),
            metrics["edge_density"], metrics["texture_variation"], metrics["colour_difference"],
            metrics["gradient_strength"], metrics["entropy"], metrics["contrast_difference"],
            metrics["mask_stability"], metrics["local_texture_contrast"], metrics["local_colour_contrast"],
            metrics["local_entropy_contrast"], metrics["internal_vs_boundary_edge_ratio"], evidence_score,
            reliability_score, metrics["coherence_score"], metrics["border_penalty"], metrics["area_reduction"],
            metrics["boundary_smoothness"], contributions, dominant, explanation, priority, mask_path,
            raw_mask_path, context_mask_path,
        ))

    combined_mask_path = MASK_DIR / f"{image_stem}_combined_mask.png"
    cv2.imwrite(str(combined_mask_path), combined_mask)
    overlay_path = OUTPUT_DIR / f"{image_stem}_{uuid4().hex[:8]}_region_proposals.png"
    cv2.imwrite(str(overlay_path), _render_overlay(image, proposals))
    comparison_paths, comparison_counts = _baseline_comparison(image, feature_maps, proposals, image_stem)
    diagnostics = ProposalDiagnostics(raw_count, filtered_count, len(candidates), threshold, tuple(p.priority.score for p in proposals))
    return ProposalResult(proposals, overlay_path, combined_mask_path, diagnostics, comparison_paths, comparison_counts)


def _fused_multiscale_mask(image: np.ndarray, fm: FeatureMaps, valid_area: np.ndarray, ablation: AblationConfig) -> tuple[np.ndarray, float, int]:
    h, w = image.shape[:2]
    sources = []
    if ablation.edge_features: sources += [fm.canny_edges, fm.sobel_gradient, fm.laplacian]
    if ablation.texture_features: sources += [fm.texture_variation, fm.lbp_texture]
    if ablation.colour_features: sources += [fm.color_variation]
    sources += [fm.threshold_mask, fm.anomaly_strength]
    votes = np.zeros((h, w), np.float32)
    for source in sources:
        threshold = float(np.percentile(source, 82 if source is not fm.threshold_mask else 65))
        binary = (source >= max(threshold, 1)).astype(np.uint8) * 255
        votes += binary.astype(np.float32) / 255.0

    tile_score = _tile_score_map(image, fm)
    fused_strength = cv2.normalize(0.58 * fm.anomaly_strength + 32.0 * votes + 85.0 * tile_score, None, 0, 255, cv2.NORM_MINMAX)
    heat_threshold = float(np.percentile(fused_strength, 82))
    minimum_votes = 2.0 if len(sources) > 3 else 1.0
    seed = ((fused_strength >= heat_threshold) & (votes >= minimum_votes) & (valid_area > 0)).astype(np.uint8) * 255
    raw_count = cv2.connectedComponents(seed)[0] - 1
    fused = np.zeros_like(seed)
    scales = (0.008, 0.018, 0.04) if ablation.multi_scale_fusion else (0.018,)
    for ratio in scales:
        k = _odd(max(3, int(min(h, w) * ratio)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        scale_mask = cv2.morphologyEx(seed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        scale_mask = cv2.morphologyEx(scale_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        fused = cv2.bitwise_or(fused, scale_mask)
    fused = cv2.bitwise_and(fused, valid_area)
    return fused, heat_threshold, raw_count


def _tile_score_map(image: np.ndarray, fm: FeatureMaps) -> np.ndarray:
    h, w = image.shape[:2]
    accum, weight = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    gray = fm.grayscale
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    global_stats = [np.mean(x) for x in (fm.canny_edges, fm.sobel_gradient, fm.laplacian, fm.texture_variation, fm.color_variation)]
    raw_tiles: list[tuple[tuple[int, int, int, int], list[float]]] = []
    for ratio in (0.08, 0.16, 0.28):
        size = max(24, int(min(h, w) * ratio)); step = max(8, size // 2)
        for y in range(0, max(h - size + 1, 1), step):
            for x in range(0, max(w - size + 1, 1), step):
                y2, x2 = min(y + size, h), min(x + size, w)
                patch = gray[y:y2, x:x2]
                hist = cv2.calcHist([patch], [0], None, [32], [0, 256]).ravel(); prob = hist / max(hist.sum(), 1)
                entropy = float(-np.sum(prob[prob > 0] * np.log2(prob[prob > 0])) / 5.0)
                ring = lab[max(0, y-size//2):min(h, y2+size//2), max(0, x-size//2):min(w, x2+size//2)]
                colour = float(np.linalg.norm(np.mean(lab[y:y2, x:x2], axis=(0, 1)) - np.mean(ring, axis=(0, 1))))
                contrast = abs(float(np.std(patch)) - float(np.std(gray[max(0,y-size//2):min(h,y2+size//2), max(0,x-size//2):min(w,x2+size//2)])))
                vals = [float(np.mean(m[y:y2, x:x2])) for m in (fm.canny_edges, fm.sobel_gradient, fm.laplacian, fm.texture_variation, fm.color_variation)]
                vals = [v / max(g * 2.5, 20.0) for v, g in zip(vals, global_stats)] + [colour / 45.0, entropy, contrast / 35.0]
                raw_tiles.append(((x, y, x2, y2), vals))
    matrix = np.asarray([v for _, v in raw_tiles], np.float32)
    matrix = np.clip(matrix, 0, np.percentile(matrix, 97, axis=0))
    matrix /= np.maximum(np.percentile(matrix, 90, axis=0), 1e-6)
    scores = np.clip(np.mean(matrix, axis=1), 0, 1)
    for ((x1, y1, x2, y2), _), score in zip(raw_tiles, scores):
        accum[y1:y2, x1:x2] += score; weight[y1:y2, x1:x2] += 1
    return accum / np.maximum(weight, 1)


def _components(mask: np.ndarray) -> list[_Candidate]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    return [_Candidate((labels == i).astype(np.uint8) * 255, (int(stats[i,0]), int(stats[i,1]), int(stats[i,0]+stats[i,2]), int(stats[i,1]+stats[i,3]))) for i in range(1, count)]


def _keep_candidate(c: _Candidate, minimum: int, image_area: int, max_relative: float, width: int, height: int, fm: FeatureMaps) -> bool:
    area = cv2.countNonZero(c.mask); x1, y1, x2, y2 = c.bbox
    if area < minimum or area / image_area > max_relative: return False
    bw, bh = x2-x1, y2-y1
    if min(bw, bh) < 3: return False
    border = x1 <= 1 or y1 <= 1 or x2 >= width-1 or y2 >= height-1
    fill = area / max(bw * bh, 1)
    if border and fill < 0.08: return False
    if area < image_area * 0.001 and np.mean(fm.color_variation[c.mask > 0]) > 150 and np.mean(fm.texture_variation[c.mask > 0]) < 25: return False
    return True


def _merge_candidates(candidates: list[_Candidate], fm: FeatureMaps, width: int, height: int) -> list[_Candidate]:
    items = candidates[:]
    changed = True
    while changed:
        changed = False
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if _should_merge(items[i], items[j], fm, width, height):
                    mask = cv2.bitwise_or(items[i].mask, items[j].mask)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(max(3, int(min(width,height)*.012))),) * 2)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                    ys, xs = np.where(mask > 0)
                    items[i] = _Candidate(mask, (int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)))
                    items.pop(j); changed = True; break
            if changed: break
    return items


def _should_merge(a: _Candidate, b: _Candidate, fm: FeatureMaps, width: int, height: int) -> bool:
    iou = _bbox_iou(a.bbox, b.bbox)
    ac = ((a.bbox[0]+a.bbox[2])/2, (a.bbox[1]+a.bbox[3])/2); bc = ((b.bbox[0]+b.bbox[2])/2, (b.bbox[1]+b.bbox[3])/2)
    distance = np.hypot(ac[0]-bc[0], ac[1]-bc[1]) / max(width, height)
    texture_delta = abs(_masked_mean(fm.texture_variation, a.mask) - _masked_mean(fm.texture_variation, b.mask)) / 255
    colour_delta = abs(_masked_mean(fm.color_variation, a.mask) - _masked_mean(fm.color_variation, b.mask)) / 255
    dilated = cv2.dilate(a.mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15)))
    connected = bool(np.any((dilated > 0) & (b.mask > 0)))
    return iou > 0.08 or (distance < 0.13 and texture_delta < .28 and colour_delta < .28) or (connected and texture_delta < .35)


def _region_metrics(image: np.ndarray, fm: FeatureMaps, mask: np.ndarray, bbox: tuple[int,int,int,int]) -> dict[str, float]:
    x1,y1,x2,y2 = bbox; selected = mask > 0
    gray_values = fm.grayscale[selected]
    hist = cv2.calcHist([fm.grayscale], [0], mask, [32], [0,256]).ravel(); p = hist/max(hist.sum(),1)
    entropy = float(-np.sum(p[p>0]*np.log2(p[p>0]))/5.0)
    ring = _context_ring(mask) > 0
    outer = fm.grayscale[ring]
    if outer.size == 0: outer = fm.grayscale[max(0,y1-20):min(fm.grayscale.shape[0],y2+20), max(0,x1-20):min(fm.grayscale.shape[1],x2+20)].ravel()
    candidate_entropy = entropy
    ring_entropy = _entropy(fm.grayscale, (ring.astype(np.uint8) * 255))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    candidate_lab = np.mean(lab[selected], axis=0); ring_lab = np.mean(lab[ring], axis=0) if np.any(ring) else candidate_lab
    texture_inside = _masked_mean(fm.texture_variation, mask); texture_ring = float(np.mean(fm.texture_variation[ring])) if np.any(ring) else texture_inside
    gradient_inside = _masked_mean(fm.sobel_gradient, mask); gradient_ring = float(np.mean(fm.sobel_gradient[ring])) if np.any(ring) else gradient_inside
    boundary = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((5,5),np.uint8))
    edge_internal = _masked_mean(fm.canny_edges, mask); edge_boundary = _masked_mean(fm.canny_edges, boundary)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    perimeter=sum(cv2.arcLength(c,True) for c in contours); area=max(cv2.countNonZero(mask),1)
    irregularity=min((perimeter * perimeter / (4*np.pi*area)-1)/8,1.0)
    area_rel = cv2.countNonZero(mask) / mask.size
    area_relevance = min(1.0, np.sqrt(area_rel / 0.08))
    return {
        "edge_density": _masked_mean(fm.canny_edges, mask)/255,
        "texture_variation": _masked_mean(fm.texture_variation, mask)/255,
        "colour_difference": _masked_mean(fm.color_variation, mask)/255,
        "gradient_strength": _masked_mean(fm.sobel_gradient, mask)/255,
        "entropy": min(entropy,1.0), "area_relevance": area_relevance,
        "contrast_difference": min(abs(float(np.std(gray_values))-float(np.std(outer)))/64,1.0),
        "local_texture_contrast": min(abs(texture_inside-texture_ring)/80,1.0),
        "local_colour_contrast": min(float(np.linalg.norm(candidate_lab-ring_lab))/60,1.0),
        "local_entropy_contrast": min(abs(candidate_entropy-ring_entropy),1.0),
        "internal_vs_boundary_edge_ratio": min(edge_internal/max(edge_boundary,1.0),2.0)/2.0,
        "gradient_contrast": min(abs(gradient_inside-gradient_ring)/100,1.0),
        "geometric_irregularity": irregularity, "novelty": min((abs(texture_inside-texture_ring)+np.linalg.norm(candidate_lab-ring_lab))/100,1.0),
    }


def _valid_image_area(image: np.ndarray, margin_ratio: float) -> np.ndarray:
    h,w=image.shape[:2]; valid=np.ones((h,w),np.uint8)*255
    margin=max(0,int(min(h,w)*margin_ratio))
    if margin: valid[:margin]=0; valid[-margin:]=0; valid[:,:margin]=0; valid[:,-margin:]=0
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    limit=max(margin,int(min(h,w)*.12))
    for index in range(limit):
        if np.mean(gray[index])<18 and np.std(gray[index])<8: valid[index]=0
        if np.mean(gray[h-1-index])<18 and np.std(gray[h-1-index])<8: valid[h-1-index]=0
        if np.mean(gray[:,index])<18 and np.std(gray[:,index])<8: valid[:,index]=0
        if np.mean(gray[:,w-1-index])<18 and np.std(gray[:,w-1-index])<8: valid[:,w-1-index]=0
    return valid


def _context_ring(mask: np.ndarray) -> np.ndarray:
    radius=_odd(max(9,int(min(mask.shape)*.035)))
    outer=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius,radius)))
    inner=cv2.dilate(mask,np.ones((3,3),np.uint8))
    return cv2.subtract(outer,inner)


def _entropy(values: np.ndarray, mask: np.ndarray) -> float:
    if cv2.countNonZero(mask)==0: return 0.0
    hist=cv2.calcHist([values],[0],mask,[32],[0,256]).ravel(); p=hist/max(hist.sum(),1)
    return float(-np.sum(p[p>0]*np.log2(p[p>0]))/5.0)


def _split_candidate(candidate: _Candidate, heatmap: np.ndarray, image_area: int) -> list[_Candidate]:
    area=cv2.countNonZero(candidate.mask)
    if area < image_area*.035: return [candidate]
    values=heatmap[candidate.mask>0]
    peak_threshold=max(float(np.percentile(values,78)),float(np.mean(values)+.8*np.std(values)),float(np.max(values)*.45))
    high=((heatmap>=peak_threshold)&(candidate.mask>0)).astype(np.uint8)*255
    high=cv2.morphologyEx(high,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    seeds=_components(high)
    seeds=[seed for seed in seeds if cv2.countNonZero(seed.mask)>max(12,area*.025)]
    if len(seeds)<2: return [candidate]
    union=np.zeros_like(candidate.mask); result=[]
    grow=_odd(max(7,int(np.sqrt(area)*.12)))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(grow,grow))
    for seed in seeds[:8]:
        part=cv2.bitwise_and(cv2.dilate(seed.mask,kernel,iterations=2),candidate.mask)
        part=cv2.bitwise_and(part,cv2.bitwise_not(union)); union=cv2.bitwise_or(union,part)
        if cv2.countNonZero(part)>20:
            ys,xs=np.where(part>0); result.append(_Candidate(part,(int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)),candidate.mask))
    coverage=cv2.countNonZero(union)/max(area,1)
    return result if len(result)>1 and coverage>.35 else [candidate]


def _refine_candidate(candidate: _Candidate, fm: FeatureMaps, minimum: int) -> _Candidate:
    raw=candidate.mask.copy(); smooth=cv2.bilateralFilter(fm.anomaly_strength,7,35,35)
    local=cv2.adaptiveThreshold(smooth,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,-2)
    threshold=float(np.percentile(smooth[raw>0],55)) if np.any(raw>0) else 255
    refined=cv2.bitwise_and(raw,((smooth>=threshold)|(local>0)).astype(np.uint8)*255)
    refined=cv2.morphologyEx(refined,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    refined=cv2.morphologyEx(refined,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    refined=_fill_holes(refined)
    components=_components(refined); kept=np.zeros_like(refined)
    for item in components:
        if cv2.countNonZero(item.mask)>=max(8,minimum//3): kept=cv2.bitwise_or(kept,item.mask)
    if cv2.countNonZero(kept)<max(8,minimum//3): kept=raw
    ys,xs=np.where(kept>0)
    return _Candidate(kept,(int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)),raw)


def _split_fragmented(candidate: _Candidate, minimum: int, image_area: int) -> list[_Candidate]:
    parts=[item for item in _components(candidate.mask) if cv2.countNonZero(item.mask)>=max(12,minimum//2)]
    if len(parts)<2 or (candidate.bbox[2]-candidate.bbox[0])*(candidate.bbox[3]-candidate.bbox[1])<image_area*.05:
        return [candidate]
    total=max(cv2.countNonZero(candidate.mask),1); largest=max(cv2.countNonZero(item.mask) for item in parts)
    if largest/total>.82: return [candidate]
    output=[]
    for part in parts[:40]:
        raw=cv2.bitwise_and(candidate.raw_mask,cv2.dilate(part.mask,np.ones((7,7),np.uint8))) if candidate.raw_mask is not None else part.mask
        output.append(_Candidate(part.mask,part.bbox,raw))
    return output or [candidate]


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    flood=mask.copy(); padded=cv2.copyMakeBorder(flood,1,1,1,1,cv2.BORDER_CONSTANT,value=0)
    cv2.floodFill(padded,None,(0,0),255); inverse=cv2.bitwise_not(padded[1:-1,1:-1])
    return cv2.bitwise_or(mask,inverse)


def _coherence_metrics(mask: np.ndarray, heatmap: np.ndarray) -> dict[str,float]:
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    area=max(cv2.countNonZero(mask),1); largest=max((cv2.contourArea(c) for c in contours),default=0)
    connectedness=largest/area
    hull_area=0.0; perimeter=0.0
    if contours:
        points=np.vstack(contours); hull_area=cv2.contourArea(cv2.convexHull(points)); perimeter=sum(cv2.arcLength(c,True) for c in contours)
    solidity=min(area/max(hull_area,1),1.0)
    boundary_smoothness=1.0-min(perimeter/max(2*np.sqrt(np.pi*area),1)-1,4)/4
    vals=heatmap[mask>0]; heat_consistency=1.0-min(float(np.std(vals))/90,1.0) if vals.size else 0
    peak_level=np.percentile(vals,85) if vals.size else 255
    peaks=cv2.connectedComponents(((heatmap>=peak_level)&(mask>0)).astype(np.uint8))[0]-1
    peak_score=1.0/min(max(peaks,1),5)
    coherence=float(np.mean([connectedness,solidity,heat_consistency,peak_score]))
    raw=mask
    return {"connectedness":connectedness,"boundary_smoothness":boundary_smoothness,"scale_agreement":0.75,
            "coherence_score":coherence,"area_reduction":0.0,"fragmentation":min(len(contours)/8,1.0)}


def _border_penalty(mask: np.ndarray, valid_area: np.ndarray) -> float:
    boundary_zone=valid_area==0; total=max(cv2.countNonZero(mask),1)
    near=cv2.dilate(boundary_zone.astype(np.uint8),np.ones((9,9),np.uint8))>0
    return float(np.count_nonzero((mask>0)&near)/total)


def _robust_calibrate(rows: list[dict[str,float]]) -> list[dict[str,float]]:
    if not rows: return []
    keys=set().union(*(row.keys() for row in rows)); result=[dict(row) for row in rows]
    for key in keys:
        values=np.asarray([row.get(key,0.0) for row in rows],float)
        median=np.median(values); q1,q3=np.percentile(values,[25,75]); iqr=q3-q1
        if iqr<1e-8:
            scaled=np.clip(values,0,1)
        else:
            scaled=np.clip(.5+(values-median)/(3*iqr),0,1)
        for output,value in zip(result,scaled): output[key]=float(value)
    return result


def _mask_stability(image: np.ndarray, mask: np.ndarray, bbox: tuple[int,int,int,int]) -> float:
    x1,y1,x2,y2 = bbox; target = mask[y1:y2,x1:x2] > 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY); rng = np.random.default_rng(7)
    variants = [cv2.convertScaleAbs(image, alpha=1.0, beta=12), cv2.convertScaleAbs(image, alpha=1.08, beta=0),
                np.clip(image.astype(np.float32)+rng.normal(0,4,image.shape),0,255).astype(np.uint8)]
    scores = []; baseline_support = _saliency_support(gray)[y1:y2,x1:x2] & target
    for variant in variants:
        support = _saliency_support(cv2.cvtColor(variant, cv2.COLOR_BGR2GRAY))[y1:y2,x1:x2] & target
        scores.append(_binary_iou(baseline_support, support))
    resized = cv2.resize(gray, None, fx=.75, fy=.75); restored = cv2.resize(resized, (gray.shape[1],gray.shape[0]))
    scores.append(_binary_iou(baseline_support, _saliency_support(restored)[y1:y2,x1:x2] & target))
    return float(np.mean(scores))


def _saliency_support(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0); gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1)
    texture = cv2.absdiff(gray, cv2.blur(gray, (15, 15))).astype(np.float32)
    saliency = cv2.magnitude(gx, gy) + texture
    return saliency >= np.percentile(saliency, 75)


def _binary_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 1.0


def _dominant_features(contributions: dict[str,float]) -> tuple[str,...]:
    names={"local_texture_contrast":"Texture contrast","local_colour_contrast":"Colour contrast",
           "local_entropy_contrast":"Entropy contrast","edge_concentration":"Edge concentration",
           "gradient_contrast":"Gradient contrast","geometric_irregularity":"Geometry"}
    return tuple(names[k] for k,_ in sorted(contributions.items(),key=lambda item:item[1],reverse=True)[:2])


def _explain(m: dict[str,float], aspect: float, stability: float, relative_area: float) -> str:
    reasons=[]
    if m["local_texture_contrast"]>.25: reasons.append("texture differs from local context")
    if m["internal_vs_boundary_edge_ratio"]>.35: reasons.append("internal edge concentration")
    if m["local_colour_contrast"]>.20: reasons.append("colour differs from local context")
    if m["local_entropy_contrast"]>.15: reasons.append("entropy differs from local context")
    if aspect>3: reasons.append("elongated geometry")
    if relative_area>.06 and stability>.7: reasons.append("large stable anomaly")
    if stability<.6: reasons.append("instability under perturbation")
    return ", ".join(reasons) if reasons else "combined multi-feature anomaly evidence"


def _render_overlay(image: np.ndarray, proposals: list[RegionProposal]) -> np.ndarray:
    overlay=image.copy(); tint=image.copy()
    for p in proposals:
        mask = cv2.imread(str(p.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            tint[mask > 0] = (0,170,255)
    overlay=cv2.addWeighted(overlay,.72,tint,.28,0)
    for p in proposals:
        x1,y1,x2,y2=p.bbox; cv2.rectangle(overlay,(x1,y1),(x2,y2),(0,210,255),2)
        label=p.region_id
        cv2.circle(overlay,(x1+18,max(18,y1+18)),17,(0,160,220),-1)
        cv2.putText(overlay,label,(x1+2,max(23,y1+23)),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
    return overlay


def _baseline_comparison(image: np.ndarray, fm: FeatureMaps, proposals: list[RegionProposal], stem: str) -> tuple[dict[str,Path],dict[str,int]]:
    masks={"Contour-only baseline":fm.contour_map, "Fixed-threshold baseline":(fm.anomaly_strength>128).astype(np.uint8)*255}
    paths={}; counts={}
    for name,mask in masks.items():
        comps=_components(mask); counts[name]=len(comps); vis=image.copy()
        for c in comps[:30]: cv2.rectangle(vis,(c.bbox[0],c.bbox[1]),(c.bbox[2],c.bbox[3]),(80,180,255),1)
        path=OUTPUT_DIR/f"{stem}_{name.lower().replace(' ','_').replace('-','_')}.png"; cv2.imwrite(str(path),vis); paths[name]=path
    raw_vis=image.copy()
    for proposal in proposals:
        raw=cv2.imread(str(proposal.raw_mask_path),cv2.IMREAD_GRAYSCALE)
        if raw is None: continue
        contours,_=cv2.findContours(raw,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(raw_vis,contours,-1,(255,150,40),2)
    paths["Multi-scale fused method"] = OUTPUT_DIR/f"{stem}_multiscale_comparison.png"
    cv2.imwrite(str(paths["Multi-scale fused method"]),raw_vis); counts["Multi-scale fused method"]=len(proposals)
    paths["Refined contextual method"] = OUTPUT_DIR/f"{stem}_contextual_comparison.png"
    cv2.imwrite(str(paths["Refined contextual method"]),_render_overlay(image,proposals)); counts["Refined contextual method"]=len(proposals)
    return paths,counts


def create_region_crops(image: np.ndarray, feature_maps: FeatureMaps, proposal: RegionProposal, padding: float=.18) -> dict[str,np.ndarray]:
    h,w=image.shape[:2]; x1,y1,x2,y2=proposal.bbox; pad=max(12,int(max(x2-x1,y2-y1)*padding))
    x1,y1,x2,y2=max(0,x1-pad),max(0,y1-pad),min(w,x2+pad),min(h,y2+pad)
    original=image[y1:y2,x1:x2].copy(); mask=cv2.imread(str(proposal.mask_path),0)[y1:y2,x1:x2]
    raw=cv2.imread(str(proposal.raw_mask_path),0)[y1:y2,x1:x2]; context=cv2.imread(str(proposal.context_mask_path),0)[y1:y2,x1:x2]
    def overlay(source,color):
        tinted=original.copy(); tinted[source>0]=color; return cv2.addWeighted(original,.65,tinted,.35,0)
    return {"Original crop":original,"Raw mask overlay":overlay(raw,(255,120,0)),"Refined mask overlay":overlay(mask,(0,170,255)),
            "Heatmap crop":feature_maps.anomaly_heatmap[y1:y2,x1:x2].copy(),"Local context ring":overlay(context,(90,220,90))}


def correct_region_mask(
    proposal: RegionProposal, bbox: tuple[int,int,int,int], mask_source: str="refined", morphology: int=0,
    remove_small: int=0, invert: bool=False, output_stem: str="corrected",
) -> tuple[Path, dict[str,float]]:
    source=proposal.raw_mask_path if mask_source=="raw" else proposal.mask_path
    mask=cv2.imread(str(source),cv2.IMREAD_GRAYSCALE)
    if mask is None: raise ValueError("Proposal mask is unavailable.")
    x1,y1,x2,y2=bbox; bounded=np.zeros_like(mask); bounded[max(0,y1):min(mask.shape[0],y2),max(0,x1):min(mask.shape[1],x2)]=255
    mask=cv2.bitwise_and(mask,bounded)
    if invert: mask=cv2.bitwise_and(cv2.bitwise_not(mask),bounded)
    if morphology:
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)); operation=cv2.dilate if morphology>0 else cv2.erode
        mask=operation(mask,kernel,iterations=abs(morphology))
    if remove_small:
        cleaned=np.zeros_like(mask)
        for item in _components(mask):
            if cv2.countNonZero(item.mask)>=remove_small: cleaned=cv2.bitwise_or(cleaned,item.mask)
        mask=cleaned
    path=MASK_DIR/f"{output_stem}_{proposal.region_id}_corrected.png"; cv2.imwrite(str(path),mask)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); area=cv2.countNonZero(mask)
    perimeter=sum(cv2.arcLength(c,True) for c in contours)
    return path,{"pixel_area":area,"components":len(contours),"boundary_smoothness":round(1-min(perimeter/max(2*np.sqrt(np.pi*max(area,1)),1)-1,4)/4,3)}


def create_region_crop(image: np.ndarray, proposal: RegionProposal) -> np.ndarray:
    x1,y1,x2,y2=proposal.bbox; pad=max(12,int(max(x2-x1,y2-y1)*.18)); h,w=image.shape[:2]
    return image[max(0,y1-pad):min(h,y2+pad),max(0,x1-pad):min(w,x2+pad)].copy()


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    selected=values[mask>0]; return float(np.mean(selected)) if selected.size else 0.0


def _bbox_iou(a: tuple[int,int,int,int], b: tuple[int,int,int,int]) -> float:
    x1,y1=max(a[0],b[0]),max(a[1],b[1]); x2,y2=min(a[2],b[2]),min(a[3],b[3]); inter=max(0,x2-x1)*max(0,y2-y1)
    return inter/max((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter,1)


def _odd(value: int) -> int: return value if value%2 else value+1

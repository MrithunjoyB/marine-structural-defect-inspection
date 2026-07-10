"""Multi-scale, explainable classical-CV anomaly region proposals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from config import MASK_DIR, OUTPUT_DIR
from feature_extraction import FeatureMaps, extract_feature_maps
from scoring import DEFAULT_SCORE_WEIGHTS, PriorityResult, priority_label, weighted_anomaly_score


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
    dominant_features: tuple[str, ...]
    explanation: str
    priority: PriorityResult
    mask_path: Path

    def to_row(self) -> dict[str, object]:
        x1, y1, x2, y2 = self.bbox
        return {
            "Region ID": self.region_id, "BBox": f"({x1}, {y1}) - ({x2}, {y2})",
            "Pixel Area": self.pixel_area, "Relative Area (%)": round(self.relative_area * 100, 3),
            "Aspect Ratio": round(self.aspect_ratio, 2), "Perimeter": round(self.perimeter, 1),
            "Edge Density": round(self.edge_density, 3), "Texture Score": round(self.texture_score, 3),
            "Color Variation": round(self.color_variation_score, 3), "Gradient": round(self.gradient_strength, 3),
            "Entropy": round(self.entropy, 3), "Stability": round(self.mask_stability, 3),
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


def propose_regions(
    image: np.ndarray, feature_maps: FeatureMaps, image_stem: str, min_area: int = 250,
    max_regions: int = 20, min_relative_area: float = 0.0002, max_relative_area: float = 0.85,
    score_weights: dict[str, float] | None = None,
) -> ProposalResult:
    """Fuse independent feature masks, overlapping tiles, and multi-scale components."""
    height, width = image.shape[:2]
    image_area = max(height * width, 1)
    min_pixels = max(16, min(min_area, int(image_area * 0.01)), int(image_area * min_relative_area))
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fused, threshold, raw_count = _fused_multiscale_mask(image, feature_maps)
    components = _components(fused)
    filtered = [c for c in components if _keep_candidate(c, min_pixels, image_area, max_relative_area, width, height, feature_maps)]
    filtered_count = len(filtered)
    merged = _merge_candidates(filtered, feature_maps, width, height)

    scored: list[tuple[float, _Candidate, dict[str, float], float]] = []
    for candidate in merged:
        metrics = _region_metrics(image, feature_maps, candidate.mask, candidate.bbox)
        stability = _mask_stability(image, candidate.mask, candidate.bbox)
        metrics["mask_stability"] = stability
        score = weighted_anomaly_score(metrics, score_weights or DEFAULT_SCORE_WEIGHTS)
        scored.append((score, candidate, metrics, stability))
    scored.sort(key=lambda item: item[0], reverse=True)
    scored = scored[:max_regions]

    proposals: list[RegionProposal] = []
    combined_mask = np.zeros((height, width), np.uint8)
    for index, (score, candidate, metrics, stability) in enumerate(scored, 1):
        region_id = f"R{index:03d}"
        mask_path = MASK_DIR / f"{image_stem}_{region_id}_mask.png"
        cv2.imwrite(str(mask_path), candidate.mask)
        combined_mask = cv2.bitwise_or(combined_mask, candidate.mask)
        x1, y1, x2, y2 = candidate.bbox
        contour_list, _ = cv2.findContours(candidate.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = int(cv2.countNonZero(candidate.mask))
        perimeter = sum(cv2.arcLength(c, True) for c in contour_list)
        aspect = max((x2 - x1) / max(y2 - y1, 1), (y2 - y1) / max(x2 - x1, 1))
        dominant = _dominant_features(metrics)
        explanation = _explain(metrics, aspect, stability, area / image_area)
        priority = PriorityResult(round(score, 1), priority_label(score), explanation)
        proposals.append(RegionProposal(
            region_id, candidate.bbox, area, area / image_area, aspect, float(perimeter),
            metrics["edge_density"], metrics["texture_variation"], metrics["colour_difference"],
            metrics["gradient_strength"], metrics["entropy"], metrics["contrast_difference"],
            stability, dominant, explanation, priority, mask_path,
        ))

    combined_mask_path = MASK_DIR / f"{image_stem}_combined_mask.png"
    cv2.imwrite(str(combined_mask_path), combined_mask)
    overlay_path = OUTPUT_DIR / f"{image_stem}_{uuid4().hex[:8]}_region_proposals.png"
    cv2.imwrite(str(overlay_path), _render_overlay(image, proposals))
    comparison_paths, comparison_counts = _baseline_comparison(image, feature_maps, proposals, image_stem)
    diagnostics = ProposalDiagnostics(raw_count, filtered_count, len(merged), threshold, tuple(p.priority.score for p in proposals))
    return ProposalResult(proposals, overlay_path, combined_mask_path, diagnostics, comparison_paths, comparison_counts)


def _fused_multiscale_mask(image: np.ndarray, fm: FeatureMaps) -> tuple[np.ndarray, float, int]:
    h, w = image.shape[:2]
    sources = [fm.canny_edges, fm.sobel_gradient, fm.laplacian, fm.texture_variation,
               fm.lbp_texture, fm.color_variation, fm.threshold_mask, fm.anomaly_strength]
    votes = np.zeros((h, w), np.float32)
    for source in sources:
        threshold = float(np.percentile(source, 82 if source is not fm.threshold_mask else 65))
        binary = (source >= max(threshold, 1)).astype(np.uint8) * 255
        votes += binary.astype(np.float32) / 255.0

    tile_score = _tile_score_map(image, fm)
    fused_strength = cv2.normalize(0.58 * fm.anomaly_strength + 32.0 * votes + 85.0 * tile_score, None, 0, 255, cv2.NORM_MINMAX)
    heat_threshold = float(np.percentile(fused_strength, 82))
    seed = ((fused_strength >= heat_threshold) & (votes >= 2.0)).astype(np.uint8) * 255
    raw_count = cv2.connectedComponents(seed)[0] - 1
    fused = np.zeros_like(seed)
    for ratio in (0.008, 0.018, 0.04):
        k = _odd(max(3, int(min(h, w) * ratio)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        scale_mask = cv2.morphologyEx(seed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        scale_mask = cv2.morphologyEx(scale_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        fused = cv2.bitwise_or(fused, scale_mask)
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
    outer = fm.grayscale[max(0,y1-20):min(fm.grayscale.shape[0],y2+20), max(0,x1-20):min(fm.grayscale.shape[1],x2+20)]
    area_rel = cv2.countNonZero(mask) / mask.size
    area_relevance = min(1.0, np.sqrt(area_rel / 0.08))
    return {
        "edge_density": _masked_mean(fm.canny_edges, mask)/255,
        "texture_variation": _masked_mean(fm.texture_variation, mask)/255,
        "colour_difference": _masked_mean(fm.color_variation, mask)/255,
        "gradient_strength": _masked_mean(fm.sobel_gradient, mask)/255,
        "entropy": min(entropy,1.0), "area_relevance": area_relevance,
        "contrast_difference": min(abs(float(np.std(gray_values))-float(np.std(outer)))/64,1.0),
    }


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


def _dominant_features(metrics: dict[str,float]) -> tuple[str,...]:
    names = {"edge_density":"Edge", "texture_variation":"Texture", "colour_difference":"Colour", "gradient_strength":"Gradient", "entropy":"Entropy", "mask_stability":"Stability"}
    return tuple(names[k] for k,_ in sorted(((k,metrics[k]) for k in names), key=lambda x:x[1], reverse=True)[:2])


def _explain(m: dict[str,float], aspect: float, stability: float, relative_area: float) -> str:
    reasons=[]
    if m["texture_variation"]>.35: reasons.append("high texture variation")
    if m["edge_density"]>.18: reasons.append("high edge concentration")
    if m["colour_difference"]>.30: reasons.append("unusual colour difference")
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
        label=f"{p.region_id} | Score {p.priority.score:.0f} | {' + '.join(p.dominant_features)}"
        cv2.putText(overlay,label,(x1,max(16,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,.48,(10,10,10),3)
        cv2.putText(overlay,label,(x1,max(16,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
    return overlay


def _baseline_comparison(image: np.ndarray, fm: FeatureMaps, proposals: list[RegionProposal], stem: str) -> tuple[dict[str,Path],dict[str,int]]:
    masks={"Contour-only baseline":fm.contour_map, "Fixed-threshold baseline":(fm.anomaly_strength>128).astype(np.uint8)*255}
    paths={}; counts={}
    for name,mask in masks.items():
        comps=_components(mask); counts[name]=len(comps); vis=image.copy()
        for c in comps[:30]: cv2.rectangle(vis,(c.bbox[0],c.bbox[1]),(c.bbox[2],c.bbox[3]),(80,180,255),1)
        path=OUTPUT_DIR/f"{stem}_{name.lower().replace(' ','_').replace('-','_')}.png"; cv2.imwrite(str(path),vis); paths[name]=path
    paths["Multi-scale fused method"] = OUTPUT_DIR/f"{stem}_multiscale_comparison.png"
    cv2.imwrite(str(paths["Multi-scale fused method"]),_render_overlay(image,proposals)); counts["Multi-scale fused method"]=len(proposals)
    return paths,counts


def create_region_crops(image: np.ndarray, feature_maps: FeatureMaps, proposal: RegionProposal, padding: float=.18) -> dict[str,np.ndarray]:
    h,w=image.shape[:2]; x1,y1,x2,y2=proposal.bbox; pad=max(12,int(max(x2-x1,y2-y1)*padding))
    x1,y1,x2,y2=max(0,x1-pad),max(0,y1-pad),min(w,x2+pad),min(h,y2+pad)
    original=image[y1:y2,x1:x2].copy(); mask=cv2.imread(str(proposal.mask_path),0)[y1:y2,x1:x2]
    tinted=original.copy(); tinted[mask>0]=(0,170,255); mask_overlay=cv2.addWeighted(original,.65,tinted,.35,0)
    return {"Original crop":original,"Mask overlay":mask_overlay,"Heatmap crop":feature_maps.anomaly_heatmap[y1:y2,x1:x2].copy()}


def create_region_crop(image: np.ndarray, proposal: RegionProposal) -> np.ndarray:
    x1,y1,x2,y2=proposal.bbox; pad=max(12,int(max(x2-x1,y2-y1)*.18)); h,w=image.shape[:2]
    return image[max(0,y1-pad):min(h,y2+pad),max(0,x1-pad):min(w,x2+pad)].copy()


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    selected=values[mask>0]; return float(np.mean(selected)) if selected.size else 0.0


def _bbox_iou(a: tuple[int,int,int,int], b: tuple[int,int,int,int]) -> float:
    x1,y1=max(a[0],b[0]),max(a[1],b[1]); x2,y2=min(a[2],b[2]),min(a[3],b[3]); inter=max(0,x2-x1)*max(0,y2-y1)
    return inter/max((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter,1)


def _odd(value: int) -> int: return value if value%2 else value+1

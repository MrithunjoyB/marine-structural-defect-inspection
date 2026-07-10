"""Per-image and dataset-level quantitative proposal evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd

from labeling import ReviewedAnnotation
from region_proposal import RegionProposal


@dataclass(frozen=True)
class EvaluationMetrics:
    image_name: str = "image"
    method: str = "proposal"
    proposal_recall_iou_010: float = 0.0
    proposal_recall_iou_025: float = 0.0
    proposal_recall_iou_050: float = 0.0
    average_best_iou: float = 0.0
    mask_dice: float = 0.0
    mask_iou: float = 0.0
    false_proposals_per_image: float = 0.0
    accepted_proposals_per_image: float = 0.0
    annotation_acceptance_rate: float = 0.0
    area_over_coverage_ratio: float = 0.0
    area_under_coverage_ratio: float = 0.0
    manual_corrections: int = 0
    estimated_review_seconds: float = 0.0

    @property
    def proposal_recall(self) -> float:
        return self.proposal_recall_iou_025

    @property
    def region_coverage(self) -> float:
        return 1.0 - self.area_under_coverage_ratio

    def to_dict(self) -> dict[str, object]:
        return {key: round(value,4) if isinstance(value,float) else value for key,value in asdict(self).items()}


def evaluate_proposals(
    proposals: list[RegionProposal], references: list[tuple[int,int,int,int]], annotations: list[ReviewedAnnotation] | None=None,
    image_count: int=1, iou_threshold: float=0.3, reference_masks: list[np.ndarray] | None=None,
    image_name: str="image", method: str="proposal",
) -> EvaluationMetrics:
    boxes=[proposal.bbox for proposal in proposals]
    masks=[]
    for proposal in proposals:
        mask=cv2.imread(str(proposal.mask_path),cv2.IMREAD_GRAYSCALE)
        if mask is not None: masks.append(mask)
    return evaluate_method(boxes,references,masks,reference_masks,annotations,image_count,image_name,method)


def evaluate_method(
    boxes: list[tuple[int,int,int,int]], references: list[tuple[int,int,int,int]], masks: list[np.ndarray] | None=None,
    reference_masks: list[np.ndarray] | None=None, annotations: list[ReviewedAnnotation] | None=None, image_count: int=1,
    image_name: str="image", method: str="proposal",
) -> EvaluationMetrics:
    best=[max((_iou(ref,box) for box in boxes),default=0.0) for ref in references]
    recalls=[sum(value>=threshold for value in best)/max(len(references),1) for threshold in (.1,.25,.5)]
    matched=sum(any(_iou(box,ref)>=.25 for ref in references) for box in boxes)
    reviewed=annotations or []; accepted=sum(item.accepted for item in reviewed)
    dice_values=[]; mask_iou_values=[]; over=[]; under=[]
    for ref_index,ref in enumerate(references):
        if reference_masks and masks and ref_index<len(reference_masks):
            ref_mask=reference_masks[ref_index]>0
            candidate=max(masks,key=lambda item:_mask_iou(item>0,ref_mask),default=np.zeros_like(reference_masks[ref_index]))>0
            intersection=np.count_nonzero(candidate&ref_mask); denominator=np.count_nonzero(candidate)+np.count_nonzero(ref_mask)
            dice_values.append(2*intersection/max(denominator,1)); mask_iou_values.append(_mask_iou(candidate,ref_mask))
            over.append(np.count_nonzero(candidate&~ref_mask)/max(np.count_nonzero(ref_mask),1))
            under.append(np.count_nonzero(ref_mask&~candidate)/max(np.count_nonzero(ref_mask),1))
        else:
            best_box=max(boxes,key=lambda box:_iou(box,ref),default=(0,0,0,0)); inter=_intersection(best_box,ref)
            ref_area=max(_area(ref),1); over.append(max(_area(best_box)-inter,0)/ref_area); under.append(max(ref_area-inter,0)/ref_area)
    review_seconds=_review_duration(reviewed)
    corrections=sum(item.mask_source not in {"raw","refined"} for item in reviewed)
    return EvaluationMetrics(image_name,method,*recalls,float(np.mean(best)) if best else 0.0,
        float(np.mean(dice_values)) if dice_values else 0.0,float(np.mean(mask_iou_values)) if mask_iou_values else 0.0,
        (len(boxes)-matched)/max(image_count,1),accepted/max(image_count,1),accepted/max(len(reviewed),1),
        float(np.mean(over)) if over else 0.0,float(np.mean(under)) if under else 0.0,corrections,review_seconds)


def evaluation_tables(rows: Iterable[EvaluationMetrics]) -> tuple[pd.DataFrame,pd.DataFrame]:
    per_image=pd.DataFrame([row.to_dict() for row in rows])
    if per_image.empty: return per_image,per_image
    numeric=per_image.select_dtypes(include="number").columns
    dataset=per_image.groupby("method",as_index=False)[numeric].mean()
    return per_image,dataset


def export_evaluation_csv(rows: Iterable[EvaluationMetrics], path: Path) -> Path:
    per_image,_=evaluation_tables(rows); path.parent.mkdir(parents=True,exist_ok=True); per_image.to_csv(path,index=False); return path


def _review_duration(annotations: list[ReviewedAnnotation]) -> float:
    if len(annotations)<2: return 0.0
    try:
        times=pd.to_datetime([item.reviewed_at for item in annotations]); return float((max(times)-min(times)).total_seconds())
    except (ValueError,TypeError): return 0.0


def _area(box): return max(box[2]-box[0],0)*max(box[3]-box[1],0)
def _intersection(a,b): return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
def _iou(a,b):
    intersection=_intersection(a,b); return intersection/max(_area(a)+_area(b)-intersection,1)
def _mask_iou(a,b):
    union=np.count_nonzero(a|b); return np.count_nonzero(a&b)/union if union else 1.0

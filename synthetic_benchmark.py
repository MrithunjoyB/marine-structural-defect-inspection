"""Controlled synthetic benchmark with known anomaly masks."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from evaluation import evaluate_method, evaluation_tables
from feature_extraction import extract_feature_maps
from region_proposal import _components, propose_regions


def generate_cases(seed: int=11) -> dict[str,tuple[np.ndarray,np.ndarray]]:
    rng=np.random.default_rng(seed); cases={}
    def base():
        texture=rng.normal(0,5,(300,500,1)); return np.clip(155+np.repeat(texture,3,axis=2),0,255).astype(np.uint8)
    image=base(); mask=np.zeros(image.shape[:2],np.uint8); cv2.line(image,(35,250),(460,45),(35,35,35),4); cv2.line(mask,(35,250),(460,45),255,9); cases["thin_crack"]=(image,mask)
    image=base(); mask=np.zeros(image.shape[:2],np.uint8); pts=np.array([[30,135],[120,115],[230,145],[350,118],[470,145],[470,180],[320,165],[180,180],[30,170]]); cv2.fillPoly(image,[pts],(85,100,115)); cv2.fillPoly(mask,[pts],255); cases["weld_disturbance"]=(image,mask)
    image=base(); mask=np.zeros(image.shape[:2],np.uint8)
    for c in [(150,130),(175,155),(205,125),(340,190)]: cv2.circle(image,c,12,(55,65,75),-1); cv2.circle(mask,c,14,255,-1)
    cases["pitting_cluster"]=(image,mask)
    image=base(); mask=np.zeros(image.shape[:2],np.uint8); cv2.rectangle(image,(150,80),(350,230),(90,140,190),-1); cv2.rectangle(mask,(150,80),(350,230),255,-1); cases["colour_only"]=(image,mask)
    image=base(); mask=np.zeros(image.shape[:2],np.uint8); patch=rng.integers(80,225,(120,170,3),dtype=np.uint8); image[90:210,165:335]=patch; mask[90:210,165:335]=255; cases["texture_only"]=(image,mask)
    cases["normal_texture"]=(base(),np.zeros((300,500),np.uint8))
    gradient=np.tile(np.linspace(75,220,500,dtype=np.uint8),(300,1)); cases["illumination_gradient"]=(cv2.cvtColor(gradient,cv2.COLOR_GRAY2BGR),np.zeros((300,500),np.uint8))
    image=base(); image[:25]=0; image[-25:]=0; cases["black_border"]=(image,np.zeros((300,500),np.uint8))
    image=base(); mask=np.zeros(image.shape[:2],np.uint8)
    for c in [(90,80),(240,160),(420,220)]: cv2.circle(image,c,7,(255,255,255),-1)
    cases["specular_highlights"]=(image,mask)
    image=base(); cases["blur"]=(cv2.GaussianBlur(image,(9,9),2.5),np.zeros((300,500),np.uint8))
    image=base(); noisy=np.clip(image.astype(float)+rng.normal(0,15,image.shape),0,255).astype(np.uint8); cases["gaussian_noise"]=(noisy,np.zeros((300,500),np.uint8))
    return cases


def run_benchmark(output_csv: Path|None=None):
    rows=[]
    for name,(image,truth) in generate_cases().items():
        fm=extract_feature_maps(image); result=propose_regions(image,fm,f"benchmark_{name}",min_area=30,max_regions=15)
        contours,_=cv2.findContours(truth,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        references=[(x,y,x+w,y+h) for x,y,w,h in (cv2.boundingRect(c) for c in contours)]
        reference_masks=[truth] if np.any(truth) else []
        raw_union=np.zeros_like(truth)
        for proposal in result.proposals:
            raw=cv2.imread(str(proposal.raw_mask_path),cv2.IMREAD_GRAYSCALE)
            if raw is not None: raw_union=cv2.bitwise_or(raw_union,raw)
        methods={
            "contour-only":fm.contour_map,
            "fixed-threshold":((fm.anomaly_strength>128).astype(np.uint8)*255),
            "multi-scale-fused":raw_union,
            "refined-contextual":cv2.imread(str(result.combined_mask_path),0),
        }
        for method,mask in methods.items():
            components=_components(mask); boxes=[c.bbox for c in components]
            rows.append(evaluate_method(boxes,references,[mask],reference_masks,image_name=name,method=method))
    per_image,dataset=evaluation_tables(rows)
    if output_csv is not None: output_csv.parent.mkdir(parents=True,exist_ok=True); per_image.to_csv(output_csv,index=False)
    return per_image,dataset


if __name__=="__main__":
    per_image,dataset=run_benchmark(Path("outputs/synthetic_benchmark.csv")); print(dataset.to_string(index=False))

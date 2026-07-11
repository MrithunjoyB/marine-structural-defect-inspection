"""Controlled synthetic benchmark with known anomaly masks."""

from __future__ import annotations

from pathlib import Path
import hashlib

import cv2
import numpy as np

from evaluation import evaluate_method, evaluation_tables
from feature_extraction import extract_feature_maps
from region_proposal import _components, propose_regions


SYNTHETIC_CATEGORIES=("thin_crack","weld_disturbance","pitting_cluster","colour_only","texture_only","normal_texture","illumination_gradient","black_border","specular_highlights","blur","gaussian_noise")


def generate_cases(seed:int=11,samples_per_category:int=3,with_parameters:bool=False):
    cases={}; parameters={}
    for category_index,category in enumerate(SYNTHETIC_CATEGORIES):
        for sample_index in range(samples_per_category):
            derived_seed=int.from_bytes(hashlib.sha256(f"{seed}:{category}:{sample_index}".encode()).digest()[:8],"big")
            rng=np.random.default_rng(derived_seed); h,w=300,500
            level=int(rng.integers(125,186)); texture_sigma=float(rng.uniform(3,11)); gradient=float(rng.uniform(-25,25))
            texture=rng.normal(0,texture_sigma,(h,w,1)); ramp=np.linspace(-gradient,gradient,w)[None,:,None]
            image=np.clip(level+np.repeat(texture+ramp,3,axis=2),0,255).astype(np.uint8); mask=np.zeros((h,w),np.uint8)
            cx=int(rng.integers(90,w-90)); cy=int(rng.integers(65,h-65)); size=int(rng.integers(12,42)); angle=float(rng.uniform(-70,70)); intensity=int(rng.integers(28,105))
            if category=="thin_crack":
                length=int(rng.integers(180,390)); dx=int(np.cos(np.deg2rad(angle))*length/2); dy=int(np.sin(np.deg2rad(angle))*length/2); thickness=int(rng.integers(2,6)); cv2.line(image,(cx-dx,cy-dy),(cx+dx,cy+dy),(intensity,)*3,thickness); cv2.line(mask,(cx-dx,cy-dy),(cx+dx,cy+dy),255,thickness+4)
            elif category=="weld_disturbance":
                axes=(int(rng.integers(90,180)),int(rng.integers(14,35))); cv2.ellipse(image,(cx,cy),axes,angle,0,360,(intensity,intensity+15,intensity+30),-1); cv2.ellipse(mask,(cx,cy),axes,angle,0,360,255,-1)
            elif category=="pitting_cluster":
                for _ in range(int(rng.integers(4,9))):
                    point=(cx+int(rng.integers(-55,56)),cy+int(rng.integers(-45,46))); radius=int(rng.integers(5,15)); cv2.circle(image,point,radius,(intensity,)*3,-1); cv2.circle(mask,point,radius+2,255,-1)
            elif category=="colour_only": cv2.ellipse(image,(cx,cy),(size*2,size),angle,0,360,(intensity,140,205),-1); cv2.ellipse(mask,(cx,cy),(size*2,size),angle,0,360,255,-1)
            elif category=="texture_only":
                x1,x2=max(0,cx-size*2),min(w,cx+size*2); y1,y2=max(0,cy-size),min(h,cy+size); image[y1:y2,x1:x2]=rng.integers(55,225,(y2-y1,x2-x1,3),dtype=np.uint8); mask[y1:y2,x1:x2]=255
            elif category=="illumination_gradient": image=np.clip(image.astype(float)+np.linspace(-55,55,w)[None,:,None],0,255).astype(np.uint8)
            elif category=="black_border": border=int(rng.integers(10,30)); image[:border]=0; image[-border:]=0
            elif category=="specular_highlights":
                for _ in range(int(rng.integers(2,6))): cv2.circle(image,(int(rng.integers(20,w-20)),int(rng.integers(20,h-20))),int(rng.integers(4,10)),(255,255,255),-1)
            elif category=="blur": image=cv2.GaussianBlur(image,(9,9),float(rng.uniform(1.5,4)))
            elif category=="gaussian_noise": image=np.clip(image.astype(float)+rng.normal(0,float(rng.uniform(10,25)),image.shape),0,255).astype(np.uint8)
            name=f"{category}_{sample_index+1:02d}"; cases[name]=(image,mask)
            parameters[name]={"master_seed":seed,"derived_seed":derived_seed,"anomaly_type":category,"sample_index":sample_index,"position":[cx,cy],"size":size,"intensity":intensity,"orientation":angle,"background_level":level,"texture_sigma":texture_sigma,"gradient":gradient}
    hashes=[]
    for image,_ in cases.values(): hashes.append(hashlib.sha256(cv2.imencode(".png",image)[1].tobytes()).hexdigest())
    assert len(hashes)==len(set(hashes)),"Synthetic generation produced an unintended exact duplicate"
    return (cases,parameters) if with_parameters else cases


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

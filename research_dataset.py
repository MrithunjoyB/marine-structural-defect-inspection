"""Research dataset registration, ingestion, validation, splitting, and provenance."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import sqlite3
import sys
from typing import Iterable
from uuid import uuid4
import zipfile
from io import BytesIO
import csv
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pandas as pd


SOURCE_TYPES=("professor-provided","public research dataset","self-captured","synthetic","local development image","other")
GROUND_TRUTH_OPTIONS=("verified expert annotation","verified dataset annotation","reviewer-estimated","no annotation","unknown")
ANNOTATION_FORMATS=("none","YOLO bounding boxes","YOLO segmentation","COCO JSON","Pascal VOC","binary masks","CSV regions","custom")
SUPPORTED_IMAGE_SUFFIXES={".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}
FINAL_STATUS="Final Research Evaluation"


@dataclass(frozen=True)
class DatasetMetadata:
    dataset_id:str; dataset_name:str; dataset_version:str; source_type:str; source_name:str
    source_reference:str; provider_author:str; licence:str; redistribution_allowed:bool
    commercial_use_allowed:bool; citation_text:str; date_acquired:str; domain_category:str
    ground_truth_status:str; annotation_format:str; notes:str=""

    def validate(self)->None:
        required=("dataset_id","dataset_name","dataset_version","source_type","source_name","provider_author","licence","date_acquired","domain_category","ground_truth_status","annotation_format")
        missing=[name for name in required if not str(getattr(self,name)).strip()]
        if missing: raise ValueError(f"Required dataset metadata is missing: {', '.join(missing)}")
        if self.source_type not in SOURCE_TYPES: raise ValueError("Unsupported source type")
        if self.ground_truth_status not in GROUND_TRUTH_OPTIONS: raise ValueError("Unsupported ground-truth status")
        if self.annotation_format not in ANNOTATION_FORMATS: raise ValueError("Unsupported annotation format")
        if any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in self.dataset_id):
            raise ValueError("dataset_id may contain only letters, digits, hyphens, and underscores")


@dataclass(frozen=True)
class ImageManifestRecord:
    image_id:str; dataset_id:str; original_filename:str; stored_filename:str; sha256_hash:str
    width:int; height:int; channels:int; file_format:str; file_size:int; source:str; licence:str
    ground_truth_status:str; annotation_path:str; split:str; duplicate_status:str
    corruption_status:str; imported_timestamp:str; notes:str=""; perceptual_hash:str=""
    group_id:str=""; primary_class:str=""


@dataclass(frozen=True)
class ValidationReport:
    total_files:int; valid_images:int; corrupted_files:int; exact_duplicates:int
    possible_near_duplicates:int; annotated_images:int; unannotated_images:int
    class_distribution:dict[str,int]; image_size_distribution:dict[str,int]
    missing_annotations:int; invalid_annotations:int

    def to_dict(self): return asdict(self)


class DatasetRegistry:
    def __init__(self,root:Path):
        self.root=Path(root); self.registry_dir=self.root/"registry"; self.db_path=self.registry_dir/"datasets.sqlite"
        for name in ("registry","raw","processed","annotations","splits","reports","exports"):(self.root/name).mkdir(parents=True,exist_ok=True)
        self._initialize()

    def connect(self):
        con=sqlite3.connect(str(self.db_path)); con.row_factory=sqlite3.Row; return con

    def _initialize(self):
        with self.connect() as con:
            con.execute("CREATE TABLE IF NOT EXISTS datasets (dataset_id TEXT, dataset_version TEXT, metadata_json TEXT NOT NULL, registered_timestamp TEXT NOT NULL, PRIMARY KEY(dataset_id,dataset_version))")
            fields=list(ImageManifestRecord.__annotations__)
            columns=", ".join(f"{field} INTEGER" if field in {"width","height","channels","file_size"} else f"{field} TEXT" for field in fields)
            con.execute(f"CREATE TABLE IF NOT EXISTS images ({columns}, PRIMARY KEY(image_id))")
            con.execute("CREATE TABLE IF NOT EXISTS reviews (review_id TEXT PRIMARY KEY,image_id TEXT,reference_type TEXT,outcome TEXT,bbox_json TEXT,mask_path TEXT,reviewer_id TEXT,confidence REAL,notes TEXT,reviewed_timestamp TEXT)")
            con.execute("CREATE TABLE IF NOT EXISTS experiment_plans (plan_id TEXT PRIMARY KEY,experiment_id TEXT,dataset_id TEXT,dataset_version TEXT,split TEXT,selected_image_ids_json TEXT,configuration_json TEXT,manifest_hash TEXT,code_commit_hash TEXT,created_timestamp TEXT,status TEXT,reviewer_id TEXT)")

    def register_dataset(self,metadata:DatasetMetadata,overwrite=False):
        metadata.validate(); now=datetime.now().isoformat(timespec="seconds")
        with self.connect() as con:
            exists=con.execute("SELECT 1 FROM datasets WHERE dataset_id=? AND dataset_version=?",(metadata.dataset_id,metadata.dataset_version)).fetchone()
            if exists and not overwrite: raise ValueError("Dataset ID and version already exist")
            con.execute("INSERT OR REPLACE INTO datasets VALUES(?,?,?,?)",(metadata.dataset_id,metadata.dataset_version,json.dumps(asdict(metadata)),now))
        self._write_manifest_json()

    def datasets(self)->pd.DataFrame:
        with self.connect() as con: rows=con.execute("SELECT * FROM datasets ORDER BY registered_timestamp DESC").fetchall()
        return pd.DataFrame([{**json.loads(r["metadata_json"]),"registered_timestamp":r["registered_timestamp"]} for r in rows])

    def metadata(self,dataset_id,version=None)->DatasetMetadata:
        query="SELECT metadata_json FROM datasets WHERE dataset_id=?"; params=[dataset_id]
        if version: query+=" AND dataset_version=?"; params.append(version)
        query+=" ORDER BY registered_timestamp DESC LIMIT 1"
        with self.connect() as con: row=con.execute(query,params).fetchone()
        if not row: raise KeyError(dataset_id)
        return DatasetMetadata(**json.loads(row[0]))

    def add_images(self,records:Iterable[ImageManifestRecord]):
        fields=list(ImageManifestRecord.__annotations__); placeholders=",".join("?" for _ in fields)
        with self.connect() as con:
            for record in records: con.execute(f"INSERT OR REPLACE INTO images ({','.join(fields)}) VALUES ({placeholders})",[getattr(record,f) for f in fields])
        self._write_manifest_json()

    def images(self,dataset_id=None)->pd.DataFrame:
        query="SELECT * FROM images"; params=[]
        if dataset_id: query+=" WHERE dataset_id=?"; params=[dataset_id]
        with self.connect() as con: rows=con.execute(query,params).fetchall()
        return pd.DataFrame([dict(r) for r in rows],columns=list(ImageManifestRecord.__annotations__))

    def hashes(self):
        with self.connect() as con: return {r[0] for r in con.execute("SELECT sha256_hash FROM images WHERE corruption_status='valid'")}

    def save_review(self,image_id,outcome,reviewer_id,confidence,notes,bbox=None,mask_bytes=None):
        if outcome not in {"anomaly present","no anomaly","uncertain"}: raise ValueError("Invalid review outcome")
        if not reviewer_id.strip(): raise ValueError("Reviewer ID is required")
        mask_path=""
        if mask_bytes:
            mask_dir=self.root/"annotations"/"references"; mask_dir.mkdir(parents=True,exist_ok=True); path=mask_dir/f"{image_id}_reference.png"; path.write_bytes(mask_bytes); mask_path=str(path)
        with self.connect() as con: con.execute("INSERT INTO reviews VALUES(?,?,?,?,?,?,?,?,?,?)",(str(uuid4()),image_id,"reference ground truth",outcome,json.dumps(bbox) if bbox else "",mask_path,reviewer_id,float(confidence),notes,datetime.now().isoformat(timespec="seconds")))

    def create_experiment_plan(self,experiment_id,dataset_id,dataset_version,split,subset_size,status,reviewer_id,methods,parameters,seed,override=False):
        metadata=self.metadata(dataset_id,dataset_version); images=self.images(dataset_id); images=images[images.split==split] if split!="all" else images
        if status==FINAL_STATUS:
            problems=[]
            if metadata.ground_truth_status in {"unknown","no annotation"}: problems.append("verified or reviewer-estimated ground truth")
            if not metadata.source_name or not metadata.provider_author or not metadata.licence: problems.append("source, provider, and licence metadata")
            if metadata.licence.lower()=="unknown": problems.append("known licence")
            if problems and not override: raise ValueError("Final experiment requires "+", ".join(problems))
        selected=images.sample(min(int(subset_size),len(images)),random_state=int(seed)) if len(images) else images
        snapshot=create_configuration_snapshot(parameters)
        manifest_hash=hashlib.sha256(selected.to_json(orient="records").encode()).hexdigest()
        plan_id=str(uuid4()); config={**snapshot,"proposal_methods":list(methods),"random_seed":int(seed)}
        with self.connect() as con: con.execute("INSERT INTO experiment_plans VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",(plan_id,experiment_id,dataset_id,dataset_version,split,json.dumps(selected.image_id.tolist()),json.dumps(config),manifest_hash,config["code_commit_hash"],datetime.now().isoformat(timespec="seconds"),status,reviewer_id))
        path=self.root/"exports"/f"{experiment_id}_configuration.json"; path.write_text(json.dumps({"plan_id":plan_id,"dataset_id":dataset_id,"dataset_version":dataset_version,"manifest_hash":manifest_hash,"selected_image_ids":selected.image_id.tolist(),"configuration":config},indent=2))
        return path

    def _write_manifest_json(self):
        payload={"datasets":self.datasets().to_dict("records"),"images":self.images().to_dict("records")}
        (self.registry_dir/"dataset_manifest.json").write_text(json.dumps(payload,indent=2,default=str))


def ingest_files(registry:DatasetRegistry,metadata:DatasetMetadata,files:Iterable[tuple[str,bytes]],annotation_files:dict[str,bytes]|None=None)->tuple[list[ImageManifestRecord],ValidationReport]:
    registry.register_dataset(metadata,overwrite=True); annotation_files=annotation_files or {}; records=[]; existing=registry.images(); existing_hashes=set(existing.sha256_hash) if not existing.empty else set(); existing_phashes=existing.perceptual_hash.tolist() if not existing.empty else []
    raw_dir=registry.root/"raw"/metadata.dataset_id; ann_dir=registry.root/"annotations"/metadata.dataset_id; raw_dir.mkdir(parents=True,exist_ok=True); ann_dir.mkdir(parents=True,exist_ok=True)
    class_counts={}; invalid_annotations=missing_annotations=0
    for original,data in files:
        suffix=Path(original).suffix.lower(); sha=hashlib.sha256(data).hexdigest(); stored=f"{sha[:12]}_{Path(original).name}"; corruption="valid"; width=height=channels=0; phash=""; duplicate="exact duplicate" if sha in existing_hashes else "unique"
        if not data or suffix not in SUPPORTED_IMAGE_SUFFIXES: corruption="zero-byte" if not data else "unsupported format"
        image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_UNCHANGED) if corruption=="valid" else None
        if image is None and corruption=="valid": corruption="corrupt"
        if image is not None:
            height,width=image.shape[:2]; channels=1 if image.ndim==2 else image.shape[2]; phash=perceptual_hash(image)
            if width<8 or height<8 or width>20000 or height>20000: corruption="extreme resolution"
            if duplicate=="unique" and any(hamming_distance(phash,p)<=5 for p in existing_phashes if p): duplicate="possible near duplicate"
            (raw_dir/stored).write_bytes(data)
        ann_path=""; primary_class=""
        candidate_names=[Path(original).stem+ext for ext in (".txt",".json",".xml",".csv",".png")]
        matched=next((name for name in candidate_names if name in annotation_files),None)
        if metadata.annotation_format!="none":
            if not matched: missing_annotations+=1
            else:
                ann_data=annotation_files[matched]; ann_target=ann_dir/matched; ann_target.write_bytes(ann_data); ann_path=str(ann_target)
                valid,classes=validate_annotation(metadata.annotation_format,ann_data,width,height)
                invalid_annotations+=int(not valid)
                if classes: primary_class=classes[0]
                for item in classes: class_counts[item]=class_counts.get(item,0)+1
        records.append(ImageManifestRecord(str(uuid4()),metadata.dataset_id,original,stored,sha,width,height,channels,suffix.lstrip("."),len(data),metadata.source_name,metadata.licence,metadata.ground_truth_status,ann_path,"unassigned",duplicate,corruption,datetime.now().isoformat(timespec="seconds"),"",phash,"",primary_class))
        existing_hashes.add(sha); existing_phashes.append(phash)
    registry.add_images(records); report=build_validation_report(records,class_counts,missing_annotations,invalid_annotations); _save_validation(registry,metadata.dataset_id,records,report); return records,report


def extract_zip(data:bytes)->tuple[list[tuple[str,bytes]],dict[str,bytes]]:
    images=[]; annotations={}
    with zipfile.ZipFile(BytesIO(data)) as archive:
        for info in archive.infolist():
            if info.is_dir() or info.file_size==0: continue
            name=Path(info.filename).name
            if not name or ".." in Path(info.filename).parts: continue
            payload=archive.read(info)
            if Path(name).suffix.lower() in SUPPORTED_IMAGE_SUFFIXES: images.append((name,payload))
            else: annotations[name]=payload
    return images,annotations


def validate_annotation(fmt,data,width,height):
    if width<=0 or height<=0:return False,[]
    try:
        if fmt in {"YOLO bounding boxes","YOLO segmentation"}:
            classes=[]
            for line in data.decode().splitlines():
                parts=line.split(); int(parts[0]); classes.append(parts[0]); coordinates=[float(v) for v in parts[1:]]
                if not coordinates or any(v<0 or v>1 for v in coordinates): return False,classes
            return True,classes
        if fmt=="binary masks":
            mask=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_GRAYSCALE); return mask is not None and np.any(mask) and mask.shape[:2]==(height,width),[]
        if fmt=="COCO JSON":
            payload=json.loads(data); annotations=payload.get("annotations",[]); ids=[a.get("id") for a in annotations]
            bounds=all(len(a.get("bbox",[]))==4 and a["bbox"][0]>=0 and a["bbox"][1]>=0 and a["bbox"][2]>=0 and a["bbox"][3]>=0 and a["bbox"][0]+a["bbox"][2]<=width and a["bbox"][1]+a["bbox"][3]<=height for a in annotations)
            return len(ids)==len(set(ids)) and bounds,[str(a.get("category_id")) for a in annotations]
        if fmt=="Pascal VOC":
            root=ET.fromstring(data); classes=[]
            for obj in root.findall("object"):
                classes.append(obj.findtext("name",default="unknown")); box=obj.find("bndbox"); values=[int(float(box.findtext(name,"-1"))) for name in ("xmin","ymin","xmax","ymax")]
                if values[0]<0 or values[1]<0 or values[2]>width or values[3]>height or values[2]<=values[0] or values[3]<=values[1]:return False,classes
            return True,classes
        if fmt=="CSV regions":
            rows=list(csv.DictReader(data.decode().splitlines())); classes=[]
            for row in rows:
                values=[float(row[name]) for name in ("x1","y1","x2","y2")]; classes.append(row.get("class","unknown"))
                if values[0]<0 or values[1]<0 or values[2]>width or values[3]>height or values[2]<=values[0] or values[3]<=values[1]:return False,classes
            return bool(rows),classes
        return bool(data),[]
    except (ValueError,UnicodeDecodeError,json.JSONDecodeError): return False,[]


def prepare_split(registry:DatasetRegistry,dataset_id,ratios=(.7,.15,.15),seed=42,override_leakage=False):
    if abs(sum(ratios)-1)>1e-6 or any(v<0 for v in ratios): raise ValueError("Split ratios must be non-negative and sum to 1")
    images=registry.images(dataset_id); valid=images[images.corruption_status=="valid"].copy(); rng=random.Random(seed)
    groups={}; representatives=[]
    for _,row in valid.sort_values("image_id").iterrows():
        key=str(row.group_id).strip() if pd.notna(row.group_id) else ""
        if key:
            if row.perceptual_hash:representatives.append((row.perceptual_hash,key))
        else:
            match=next((item for item in representatives if row.perceptual_hash and hamming_distance(row.perceptual_hash,item[0])<=5),None)
            if match:key=match[1]
            else:key=row.sha256_hash; representatives.append((row.perceptual_hash,key))
        group=groups.setdefault(key,{"ids":[],"class":row.primary_class or "__unlabelled__"}); group["ids"].append(row.image_id)
    buckets={}
    for key,value in groups.items():buckets.setdefault(value["class"],[]).append(key)
    assignments={}
    for class_name,keys in sorted(buckets.items()):
        rng.shuffle(keys); class_total=sum(len(groups[key]["ids"]) for key in keys); targets=[round(class_total*ratios[0]),round(class_total*ratios[1])]; counts=[0,0,0]
        for key in keys:
            index=0 if counts[0]<targets[0] else (1 if counts[1]<targets[1] else 2)
            for image_id in groups[key]["ids"]: assignments[image_id]=("train","validation","test")[index]
            counts[index]+=len(groups[key]["ids"])
    with registry.connect() as con:
        for image_id,split in assignments.items(): con.execute("UPDATE images SET split=? WHERE image_id=?",(split,image_id))
    registry._write_manifest_json(); updated=registry.images(dataset_id); leaks=check_leakage(updated)
    if any(leaks.values()) and not override_leakage: raise ValueError(f"Split leakage detected: {leaks}")
    split_dir=registry.root/"splits"/dataset_id; split_dir.mkdir(parents=True,exist_ok=True); path=split_dir/"split_manifest.json"; path.write_text(updated.to_json(orient="records",indent=2)); return updated,leaks,path


def check_leakage(images:pd.DataFrame):
    def conflicts(column):
        if column not in images:return 0
        grouped=images[images[column].astype(str)!=""].groupby(column).split.nunique(); return int((grouped>1).sum())
    near=0; rows=images[images.perceptual_hash.astype(str)!=""]
    values=rows[["perceptual_hash","split"]].to_records(index=False)
    for index,(left_hash,left_split) in enumerate(values):
        if any(left_split!=right_split and hamming_distance(left_hash,right_hash)<=5 for right_hash,right_split in values[index+1:]):near+=1
    return {"duplicate_hash_across_splits":conflicts("sha256_hash"),"near_duplicate_across_splits":near,"group_across_splits":conflicts("group_id")}


def licence_allows_public_export(metadata:DatasetMetadata,override=False):
    unknown=not metadata.licence.strip() or metadata.licence.lower()=="unknown"
    allowed=metadata.redistribution_allowed and not unknown
    if not allowed and not override: return False,"Unknown or restricted licence: explicit override is required for a public export."
    return True,"Licence override recorded." if not allowed else "Licence metadata permits redistribution."


def create_configuration_snapshot(parameters:dict):
    try:
        import importlib.metadata as md
        packages={name:md.version(name) for name in ("streamlit","opencv-python","numpy","pandas","matplotlib")}
    except Exception: packages={}
    try:
        import subprocess
        commit=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True,check=False).stdout.strip()
    except Exception: commit="unknown"
    return {"preprocessing_settings":parameters.get("preprocessing",{}),"proposal_settings":parameters.get("proposal",{}),"feature_weights":parameters.get("feature_weights",{}),"thresholds":parameters.get("thresholds",{}),"border_margin":parameters.get("border_margin"),"maximum_regions":parameters.get("maximum_regions"),"ablation_switches":parameters.get("ablation",{}),"code_commit_hash":commit or "unknown","python_version":sys.version,"package_versions":packages,"operating_system":platform.platform(),"created_timestamp":datetime.now().isoformat(timespec="seconds")}


def register_synthetic_benchmark(registry:DatasetRegistry,dataset_id="synthetic-controlled",version="1.0",seed=11):
    from synthetic_benchmark import generate_cases
    metadata=DatasetMetadata(dataset_id,"Controlled Synthetic Benchmark",version,"synthetic","StructVision generator","local generator","StructVision-AI","MIT-compatible generated data",True,True,"Generated by synthetic_benchmark.py",date.today().isoformat(),"structural surface anomalies","verified expert annotation","binary masks",f"seed={seed}")
    files=[]; annotations={}
    for name,(image,mask) in generate_cases(seed).items():
        _,encoded=cv2.imencode(".png",image); _,mask_encoded=cv2.imencode(".png",mask); filename=f"{name}.png"; files.append((filename,encoded.tobytes())); annotations[filename]=mask_encoded.tobytes()
    records,report=ingest_files(registry,metadata,files,annotations)
    params=registry.root/"reports"/dataset_id/"generation_parameters.json"; params.parent.mkdir(parents=True,exist_ok=True); params.write_text(json.dumps({"seed":seed,"cases":[r.original_filename for r in records]},indent=2)); return records,report


def perceptual_hash(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image; resized=cv2.resize(gray,(9,8)); bits=resized[:,1:]>resized[:,:-1]; return "".join("1" if v else "0" for v in bits.flat)
def hamming_distance(left,right): return sum(a!=b for a,b in zip(left,right))+abs(len(left)-len(right))
def sha256_file(path):
    digest=hashlib.sha256()
    with open(path,"rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()


def build_validation_report(records,class_counts=None,missing_annotations=0,invalid_annotations=0):
    valid=sum(r.corruption_status=="valid" for r in records); annotated=sum(bool(r.annotation_path) for r in records); sizes={}
    for r in records:
        key=f"{r.width}x{r.height}"; sizes[key]=sizes.get(key,0)+1
    return ValidationReport(len(records),valid,len(records)-valid,sum(r.duplicate_status=="exact duplicate" for r in records),sum(r.duplicate_status=="possible near duplicate" for r in records),annotated,len(records)-annotated,class_counts or {},sizes,missing_annotations,invalid_annotations)
def _save_validation(registry,dataset_id,records,report):
    directory=registry.root/"reports"/dataset_id; directory.mkdir(parents=True,exist_ok=True); pd.DataFrame([asdict(r) for r in records]).to_csv(directory/"validation_records.csv",index=False); (directory/"validation_report.json").write_text(json.dumps(report.to_dict(),indent=2))

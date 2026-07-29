"""Execution and automatic ground-truth evaluation for registered datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import sqlite3
import time
from uuid import uuid4

import cv2
import numpy as np
import pandas as pd

from structvision.operational_storage import OperationalStorageContext

from experiment_tracking import METHOD_NAMES
from feature_extraction import extract_feature_maps
from region_proposal import _components, propose_regions


EXECUTION_STATUSES=("planned","running","completed","partially_completed","failed","cancelled")
AUTOMATIC_REVIEW_STATUS="automatically_evaluated"


class ExternalRegisteredExperimentExecutionDisabledError(RuntimeError):
    """The legacy write-oriented executor was selected in external mode."""


@dataclass(frozen=True)
class AutomaticResult:
    result_id:str; plan_id:str; experiment_id:str; experiment_version:int; image_id:str
    image_filename:str; method:str; review_status:str; run_status:str; final_proposals:int
    first_true_anomaly_proposal_rank:int|None; top_1_hit:bool|None; top_3_hit:bool|None
    top_5_hit:bool|None; top_8_hit:bool|None; true_positive_proposals:int
    false_positive_proposals:int; false_negative_anomalies:int; proposal_precision:float
    proposal_recall:float|None; mean_iou:float; best_iou:float; processing_time_seconds:float
    visualization_path:str; proposal_details_json:str; error_message:str; recorded_timestamp:str


class RegisteredExperimentStore:
    def __init__(self,path:Path): self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True); self._initialize()
    def connect(self): con=sqlite3.connect(str(self.path)); con.row_factory=sqlite3.Row; return con
    def _initialize(self):
        fields=[]
        ints={"experiment_version","final_proposals","first_true_anomaly_proposal_rank","top_1_hit","top_3_hit","top_5_hit","top_8_hit","true_positive_proposals","false_positive_proposals","false_negative_anomalies"}
        reals={"proposal_precision","proposal_recall","mean_iou","best_iou","processing_time_seconds"}
        for name in AutomaticResult.__annotations__: fields.append(f"{name} {'INTEGER' if name in ints else ('REAL' if name in reals else 'TEXT')}{' PRIMARY KEY' if name=='result_id' else ''}")
        with self.connect() as con:
            con.execute(f"CREATE TABLE IF NOT EXISTS automatic_results ({','.join(fields)}, UNIQUE(plan_id,experiment_version,image_id,method))")
            con.execute("CREATE TABLE IF NOT EXISTS executions (plan_id TEXT,experiment_version INTEGER,status TEXT,completed_pairs INTEGER,total_pairs INTEGER,started_at TEXT,updated_at TEXT,error_message TEXT,PRIMARY KEY(plan_id,experiment_version))")
    def dataframe(self,plan_id=None,experiment_version=None):
        query="SELECT * FROM automatic_results"; clauses=[]; params=[]
        if plan_id: clauses.append("plan_id=?"); params.append(plan_id)
        if experiment_version: clauses.append("experiment_version=?"); params.append(experiment_version)
        if clauses: query+=" WHERE "+" AND ".join(clauses)
        with self.connect() as con: rows=con.execute(query+" ORDER BY recorded_timestamp",params).fetchall()
        return pd.DataFrame([dict(row) for row in rows],columns=list(AutomaticResult.__annotations__))
    def save(self,result:AutomaticResult,overwrite=False):
        fields=list(AutomaticResult.__annotations__); values=[_sqlite(getattr(result,key)) for key in fields]
        sql="INSERT OR REPLACE" if overwrite else "INSERT"
        with self.connect() as con: con.execute(f"{sql} INTO automatic_results ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})",values)
    def completed_pairs(self,plan_id,version):
        frame=self.dataframe(plan_id,version); return set(zip(frame[frame.run_status=="completed"].image_id,frame[frame.run_status=="completed"].method)) if not frame.empty else set()
    def failed_pairs(self,plan_id,version):
        frame=self.dataframe(plan_id,version); return set(zip(frame[frame.run_status=="failed"].image_id,frame[frame.run_status=="failed"].method)) if not frame.empty else set()
    def set_execution(self,plan_id,version,status,completed,total,error=""):
        now=datetime.now().isoformat(timespec="seconds")
        with self.connect() as con:
            old=con.execute("SELECT started_at FROM executions WHERE plan_id=? AND experiment_version=?",(plan_id,version)).fetchone(); started=old[0] if old else now
            con.execute("INSERT OR REPLACE INTO executions VALUES(?,?,?,?,?,?,?,?)",(plan_id,version,status,completed,total,started,now,error))
    def execution(self,plan_id,version):
        with self.connect() as con: row=con.execute("SELECT * FROM executions WHERE plan_id=? AND experiment_version=?",(plan_id,version)).fetchone()
        return dict(row) if row else {"status":"planned","completed_pairs":0,"total_pairs":0}
    def next_version(self,plan_id):
        with self.connect() as con: row=con.execute("SELECT MAX(experiment_version) FROM executions WHERE plan_id=?",(plan_id,)).fetchone()
        return int(row[0] or 0)+1
    def delete_results(self,plan_id,version):
        with self.connect() as con: count=con.execute("DELETE FROM automatic_results WHERE plan_id=? AND experiment_version=?",(plan_id,version)).rowcount; con.execute("DELETE FROM executions WHERE plan_id=? AND experiment_version=?",(plan_id,version))
        return count


def load_plan(registry,plan_id):
    with registry.connect() as con: row=con.execute("SELECT * FROM experiment_plans WHERE plan_id=?",(plan_id,)).fetchone()
    if not row: raise KeyError(plan_id)
    plan=dict(row); plan["selected_image_ids"]=json.loads(plan.pop("selected_image_ids_json")); plan["configuration"]=json.loads(plan.pop("configuration_json")); return plan


def selected_images(registry,plan):
    images=registry.images(plan["dataset_id"]); selected=images[images.image_id.isin(plan["selected_image_ids"])].copy()
    order={value:index for index,value in enumerate(plan["selected_image_ids"])}; selected["_order"]=selected.image_id.map(order); return selected.sort_values("_order")


def load_ground_truth(row,path_resolver=None):
    if not row.annotation_path:return np.zeros((int(row.height),int(row.width)),np.uint8)
    selected_path=Path(str(row.annotation_path))
    if path_resolver is not None:
        resolution=path_resolver.resolve_registry_annotation(str(row.annotation_path))
        if not resolution.available or resolution.resolved_path is None:
            raise ValueError(
                "Ground-truth path resolution "
                f"{resolution.status.value}: {resolution.reason}"
            )
        selected_path=resolution.resolved_path
    mask=cv2.imread(str(selected_path),cv2.IMREAD_GRAYSCALE)
    if mask is None: raise ValueError(f"Ground-truth mask cannot be loaded: {selected_path}")
    return (mask>0).astype(np.uint8)*255


def mask_iou(left,right):
    intersection=np.count_nonzero((left>0)&(right>0)); union=np.count_nonzero((left>0)|(right>0)); return intersection/union if union else 0.0


def match_proposals(proposal_masks,truth_mask,iou_threshold=.1,mask_overlap_threshold=.25,centroid_fallback=True):
    truth_components=[item.mask for item in _components(truth_mask)]; matches=[]; matched_truth=set(); details=[]
    for rank,proposal in enumerate(proposal_masks,1):
        best_iou=0.; best_index=None; matched=False
        for index,truth in enumerate(truth_components):
            intersection=np.count_nonzero((proposal>0)&(truth>0)); iou=mask_iou(proposal,truth); overlap=intersection/max(np.count_nonzero(truth),1)
            moments=cv2.moments(proposal); inside=False
            if centroid_fallback and moments["m00"]:
                cx=int(moments["m10"]/moments["m00"]); cy=int(moments["m01"]/moments["m00"]); inside=0<=cy<truth.shape[0] and 0<=cx<truth.shape[1] and truth[cy,cx]>0
            if iou>best_iou: best_iou=iou; best_index=index
            if iou>=iou_threshold or overlap>=mask_overlap_threshold or inside: matched=True; best_index=index; best_iou=max(best_iou,iou); break
        if matched and best_index is not None: matched_truth.add(best_index)
        matches.append(matched); details.append({"rank":rank,"matched":matched,"iou":best_iou})
    first=next((index for index,value in enumerate(matches,1) if value),None); positive=bool(truth_components); ious=[item["iou"] for item in details if item["matched"]]
    return {"first_true_anomaly_proposal_rank":first,"top_1_hit":first<=1 if first else (False if positive else None),"top_3_hit":first<=3 if first else (False if positive else None),"top_5_hit":first<=5 if first else (False if positive else None),"top_8_hit":first<=8 if first else (False if positive else None),"true_positive_proposals":sum(matches),"false_positive_proposals":len(matches)-sum(matches),"false_negative_anomalies":len(truth_components)-len(matched_truth),"proposal_precision":sum(matches)/len(matches) if matches else 0.,"proposal_recall":len(matched_truth)/len(truth_components) if positive else None,"mean_iou":float(np.mean(ious)) if ious else 0.,"best_iou":max(ious,default=0.),"details":details}


def method_masks(method,feature_maps,proposal_result):
    if method=="contour-only baseline": return [item.mask for item in _components(feature_maps.contour_map)][:8]
    if method=="fixed-threshold baseline": return [item.mask for item in _components((feature_maps.anomaly_strength>128).astype(np.uint8)*255)][:8]
    if method=="multi-scale fused method":
        return [cv2.imread(str(item.raw_mask_path),cv2.IMREAD_GRAYSCALE) for item in proposal_result.proposals]
    if method=="refined contextual method": return [cv2.imread(str(item.mask_path),cv2.IMREAD_GRAYSCALE) for item in proposal_result.proposals]
    raise ValueError(f"Unsupported proposal method: {method}")


def render_matches(image,truth,proposal_masks,details,path):
    overlay=image.copy(); tinted=overlay.copy(); tinted[truth>0]=(40,220,40); blended=cv2.addWeighted(overlay,.45,tinted,.55,0); overlay[truth>0]=blended[truth>0]
    for mask,item in zip(proposal_masks,details):
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); colour=(30,210,50) if item["matched"] else (40,40,230); cv2.drawContours(overlay,contours,-1,colour,2)
        if contours:
            x,y,_,_=cv2.boundingRect(max(contours,key=cv2.contourArea)); cv2.putText(overlay,f"R{item['rank']:03d} IoU {item['iou']:.2f}",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,.4,colour,1)
    Path(path).parent.mkdir(parents=True,exist_ok=True); cv2.imwrite(str(path),overlay)


def execute_plan(registry,store,plan_id,version=1,iou_threshold=.1,mask_overlap_threshold=.25,mode="resume",progress=None,cancel=None,ablation_config=None,methods_override=None,operational_storage=None):
    if operational_storage is None:
        operational_storage = OperationalStorageContext.discover()
    if getattr(operational_storage, "is_external", False):
        raise ExternalRegisteredExperimentExecutionDisabledError(
            "External registered-experiment execution is intentionally disabled; "
            "use the future API-based benchmark runner. Read-only evidence inspection "
            "remains available."
        )
    plan=load_plan(registry,plan_id); images=selected_images(registry,plan); methods=list(methods_override or plan["configuration"].get("proposal_methods",list(METHOD_NAMES))); pairs=[(row,method) for _,row in images.iterrows() for method in methods]; completed=store.completed_pairs(plan_id,version); failed=store.failed_pairs(plan_id,version)
    if mode=="cancel" and completed: raise ValueError("Completed image-method pairs already exist; choose resume, overwrite, or create a new version")
    if mode=="retry_failed": pairs=[pair for pair in pairs if (pair[0].image_id,pair[1]) in failed]
    elif mode=="resume": pairs=[pair for pair in pairs if (pair[0].image_id,pair[1]) not in completed]
    elif mode=="overwrite": pass
    total=len(images)*len(methods); done=len(completed) if mode=="resume" else 0; started=time.perf_counter(); store.set_execution(plan_id,version,"running",done,total)
    cache={}; errors=0; cancelled=False
    for row,method in pairs:
        if cancel and cancel(): store.set_execution(plan_id,version,"cancelled",done,total); cancelled=True; break
        pair_start=time.perf_counter()
        try:
            cache_key=(row.image_id,method if ablation_config is not None else "shared")
            if cache_key not in cache:
                image=cv2.imread(str(registry.root/"raw"/row.dataset_id/row.stored_filename)); truth=load_ground_truth(row); maps=extract_feature_maps(image); stem=f"registered_{plan_id[:8]}_{row.image_id[:8]}_{str(method)[:12].replace(' ','_')}"; result=propose_regions(image,maps,stem,max_regions=int(plan["configuration"].get("maximum_regions") or 8),ablation=ablation_config)
                cache[cache_key]=(image,truth,maps,result)
            image,truth,maps,result=cache[cache_key]; evaluation_method="refined contextual method" if ablation_config is not None else method; masks=method_masks(evaluation_method,maps,result); metrics=match_proposals(masks,truth,iou_threshold,mask_overlap_threshold); visual=registry.root/"reports"/row.dataset_id/"experiments"/plan["experiment_id"]/f"v{version}_{row.image_id}_{method.replace(' ','_')}.png"; render_matches(image,truth,masks,metrics["details"],visual)
            record=AutomaticResult(str(uuid4()),plan_id,plan["experiment_id"],version,row.image_id,row.original_filename,method,AUTOMATIC_REVIEW_STATUS,"completed",len(masks),metrics["first_true_anomaly_proposal_rank"],metrics["top_1_hit"],metrics["top_3_hit"],metrics["top_5_hit"],metrics["top_8_hit"],metrics["true_positive_proposals"],metrics["false_positive_proposals"],metrics["false_negative_anomalies"],metrics["proposal_precision"],metrics["proposal_recall"],metrics["mean_iou"],metrics["best_iou"],time.perf_counter()-pair_start,str(visual),json.dumps(metrics["details"]),"",datetime.now().isoformat(timespec="seconds")); store.save(record,overwrite=mode in {"overwrite","retry_failed"}); done+=1
        except Exception as error:
            errors+=1; record=AutomaticResult(str(uuid4()),plan_id,plan["experiment_id"],version,row.image_id,row.original_filename,method,AUTOMATIC_REVIEW_STATUS,"failed",0,None,None,None,None,None,0,0,0,0.,None,0.,0.,time.perf_counter()-pair_start,"","[]",str(error),datetime.now().isoformat(timespec="seconds")); store.save(record,overwrite=True)
        if progress: progress({"current_image":row.original_filename,"current_method":method,"completed":done,"total":total,"elapsed":time.perf_counter()-started,"estimated_remaining":((time.perf_counter()-started)/max(done,1))*(total-done)})
        store.set_execution(plan_id,version,"running",done,total)
    status="cancelled" if cancelled else ("completed" if done>=total and not errors else ("partially_completed" if done else "failed")); store.set_execution(plan_id,version,status,done,total,str(errors) if errors else ""); return store.dataframe(plan_id,version)


def method_summary(results):
    completed=results[results.run_status=="completed"].copy()
    if completed.empty:return pd.DataFrame()
    rows=[]
    for method,group in completed.groupby("method"):
        positive=group[group.proposal_recall.notna()]
        rows.append({"method":method,"images":len(group),"top_1_proposal_recall":positive.top_1_hit.mean() if len(positive) else np.nan,"top_3_proposal_recall":positive.top_3_hit.mean() if len(positive) else np.nan,"top_5_proposal_recall":positive.top_5_hit.mean() if len(positive) else np.nan,"top_8_proposal_recall":positive.top_8_hit.mean() if len(positive) else np.nan,"proposal_precision":group.proposal_precision.mean(),"proposal_recall":positive.proposal_recall.mean() if len(positive) else np.nan,"false_proposals_per_image":group.false_positive_proposals.mean(),"processing_time_seconds":group.processing_time_seconds.mean()})
    return pd.DataFrame(rows)


def pairing_audit(results,expected_image_ids,expected_methods):
    """Audit an experiment matrix without changing or repairing stored rows."""
    expected={(str(image_id),str(method)) for image_id in expected_image_ids for method in expected_methods}
    observed=[(str(row.image_id),str(row.method)) for _,row in results.iterrows()]
    observed_set=set(observed)
    return {
        "expected_rows":len(expected),
        "actual_rows":len(observed),
        "unique_pairs":len(observed_set),
        "missing_pairs":sorted(expected-observed_set),
        "unexpected_pairs":sorted(observed_set-expected),
        "duplicate_pair_count":len(observed)-len(observed_set),
        "failed_rows":int((results.run_status!="completed").sum()) if "run_status" in results else 0,
        "complete":observed_set==expected and len(observed)==len(expected) and ("run_status" not in results or bool((results.run_status=="completed").all())),
    }


def _sqlite(value):
    if value is None:return None
    if isinstance(value,(bool,np.bool_)):return int(value)
    if isinstance(value,np.generic):return value.item()
    return value

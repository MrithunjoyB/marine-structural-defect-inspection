"""Reproducible ablation configurations over the existing proposal pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from region_proposal import AblationConfig
from registered_experiment import execute_plan, method_summary
from research_dataset import create_configuration_snapshot


@dataclass(frozen=True)
class AblationDefinition:
    configuration_id:str; name:str; disabled_components:tuple[str,...]; config:AblationConfig


FULL=AblationConfig()
RERANK_ONLY=replace(FULL,multi_scale_fusion=False)
ABLATION_CONFIGS=(
    AblationDefinition("ABL-FULL","Full refined contextual method",(),FULL),
    AblationDefinition("ABL-NO-TEXTURE","Without local texture context",("local_texture_context",),replace(FULL,local_texture_context=False)),
    AblationDefinition("ABL-NO-COLOUR","Without local colour context",("local_colour_context",),replace(FULL,local_colour_context=False)),
    AblationDefinition("ABL-NO-ENTROPY","Without local entropy context",("local_entropy_context",),replace(FULL,local_entropy_context=False)),
    AblationDefinition("ABL-NO-STABILITY","Without mask stability",("stability",),replace(FULL,stability=False)),
    AblationDefinition("ABL-NO-BOUNDARY-EDGE","Without internal/boundary-edge evidence",("internal_boundary_edge",),replace(FULL,internal_boundary_edge=False)),
    AblationDefinition("ABL-NO-BORDER","Without border penalty",("border_penalty",),replace(FULL,border_penalty=False)),
    AblationDefinition("ABL-NO-COHERENCE","Without coherence term",("coherence_term",),replace(FULL,coherence_term=False)),
    AblationDefinition("ABL-FUSED-ONLY","Fused features without contextual reranking",("contextual_contrast",),replace(FULL,contextual_contrast=False)),
    AblationDefinition("ABL-RERANK-ONLY","Contextual reranking only",("multi_scale_fusion",),RERANK_ONLY),
    AblationDefinition("ABL-RERANK-SPECULAR-SUPPRESS","Reranking with specular suppression",("multi_scale_fusion",),replace(RERANK_ONLY,specular_suppression=True)),
    AblationDefinition("ABL-MINIMAL","Minimal baseline configuration",("contextual_contrast","stability","multi_scale_fusion","mask_refinement"),replace(FULL,contextual_contrast=False,stability=False,multi_scale_fusion=False,mask_refinement=False)),
)
CONFIG_BY_ID={item.configuration_id:item for item in ABLATION_CONFIGS}


def configuration_snapshot(definition,experiment_id,version,seed,manifest_hash,matching,parameters=None):
    payload={"configuration_id":definition.configuration_id,"name":definition.name,"enabled_components":{key:value for key,value in asdict(definition.config).items()},"disabled_components":list(definition.disabled_components),"thresholds_and_feature_weights":parameters or {},"random_seed":int(seed),"dataset_manifest_hash":manifest_hash,"experiment_id":experiment_id,"experiment_version":int(version),"matching_thresholds":matching,"runtime":create_configuration_snapshot(parameters or {})}
    payload["snapshot_hash"]=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(); return payload


def validate_comparison(snapshots,allow_seed_difference=False):
    keys=("dataset_manifest_hash","matching_thresholds")+(tuple() if allow_seed_difference else ("random_seed",))
    for key in keys:
        if len({json.dumps(item[key],sort_keys=True) for item in snapshots})>1: raise ValueError(f"Ablation comparison requires identical {key}")
    return True


def save_ablation_plan(path,source_plan,definitions,experiment_id,version,reviewer,seed,matching,manifest_hash,parameters=None):
    snapshots=[configuration_snapshot(item,experiment_id,version,seed,manifest_hash,matching,parameters) for item in definitions]; validate_comparison(snapshots)
    payload={"source_plan_id":source_plan["plan_id"],"dataset_id":source_plan["dataset_id"],"dataset_version":source_plan["dataset_version"],"split":source_plan["split"],"selected_image_ids":source_plan["selected_image_ids"],"experiment_id":experiment_id,"experiment_version":version,"reviewer":reviewer,"configurations":snapshots}
    Path(path).parent.mkdir(parents=True,exist_ok=True); Path(path).write_text(json.dumps(payload,indent=2)); return payload


def execute_ablation_plan(registry,store,source_plan_id,definitions,version,iou_threshold,overlap_threshold,mode="resume",progress=None,experiment_id=None):
    frames=[]
    for definition in definitions:
        execute_plan(registry,store,source_plan_id,version,iou_threshold,overlap_threshold,mode,progress,ablation_config=definition.config,methods_override=[definition.configuration_id])
        if experiment_id:
            with store.connect() as con: con.execute("UPDATE automatic_results SET experiment_id=? WHERE plan_id=? AND experiment_version=? AND method=?",(experiment_id,source_plan_id,version,definition.configuration_id))
        frame=store.dataframe(source_plan_id,version); frames.append(frame[frame.method==definition.configuration_id])
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()


def ablation_leaderboard(results,weights=None):
    summary=method_summary(results); weights=weights or {"precision":.25,"recall":.25,"top_1":.2,"iou":.2,"false_proposals":.1}
    if summary.empty:return summary
    false_max=max(summary.false_proposals_per_image.max(),1); summary["balanced_score"]=weights["precision"]*summary.proposal_precision.fillna(0)+weights["recall"]*summary.proposal_recall.fillna(0)+weights["top_1"]*summary.top_1_proposal_recall.fillna(0)+weights["iou"]*results.groupby("method").mean_iou.mean().reindex(summary.method).to_numpy()+weights["false_proposals"]*(1-summary.false_proposals_per_image/false_max)
    return summary.sort_values("balanced_score",ascending=False)


def contribution_table(leaderboard):
    if leaderboard.empty or "ABL-FULL" not in set(leaderboard.method):return pd.DataFrame()
    baseline=leaderboard.set_index("method").loc["ABL-FULL"]; metrics=["proposal_precision","proposal_recall","top_1_proposal_recall","false_proposals_per_image","processing_time_seconds"]
    return pd.DataFrame([{"configuration_id":row.method,**{f"difference_{metric}":row[metric]-baseline[metric] for metric in metrics}} for _,row in leaderboard.iterrows() if row.method!="ABL-FULL"])

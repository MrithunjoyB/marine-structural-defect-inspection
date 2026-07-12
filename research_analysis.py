"""Read-only filtering, category analysis, paired comparisons, and statistics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


SEARCH_COLUMNS=("experiment_id","plan_id","dataset_id","dataset_version","dataset_split","image_filename","image_id","anomaly_type","clean_artefact_type","image_outcome","method","review_status","run_status")
TOP_K=(1,3,5,8)
ADVANCED_METHODS=("multi-scale fused method","refined contextual method")
PAIR_KEYS=("experiment_id","experiment_version","image_id")


class PairingIntegrityError(ValueError):pass


def enrich_results(results:pd.DataFrame,registry)->pd.DataFrame:
    if results.empty:return results.copy()
    with registry.connect() as con: plans=pd.DataFrame([dict(row) for row in con.execute("SELECT plan_id,dataset_id,dataset_version,split,configuration_json FROM experiment_plans")])
    images=registry.images(); frame=results.merge(plans,on="plan_id",how="left").rename(columns={"split":"dataset_split"})
    columns=["image_id","anomaly_type","image_outcome","annotation_path","stored_filename"]
    frame=frame.merge(images[columns],on="image_id",how="left"); frame["clean_artefact_type"]=np.where(frame.image_outcome=="no_anomaly",frame.anomaly_type,"")
    return frame


def filter_results(frame:pd.DataFrame,search="",filters=None,numeric=None,quick="",selected_experiment="",selected_image="",sort_column="recorded_timestamp",ascending=False,secondary=""):
    result=frame.copy(); filters=filters or {}; numeric=numeric or {}
    if search.strip():
        needle=search.strip().lower(); text=result[[c for c in SEARCH_COLUMNS if c in result]].fillna("").astype(str).agg(" ".join,axis=1).str.lower(); result=result[text.str.contains(needle,regex=False)]
    for column,values in filters.items():
        values=list(values) if isinstance(values,(list,tuple,set)) else ([values] if values not in (None,"") else [])
        if values and column in result: result=result[result[column].isin(values)]
    for column,(minimum,maximum) in numeric.items():
        if column not in result:continue
        values=pd.to_numeric(result[column],errors="coerce"); result=result[(minimum is None or values.ge(minimum))&(maximum is None or values.le(maximum))]
    quick_map={
        "Current selected experiment":lambda f:f[f.experiment_id==selected_experiment],"SYN-BALANCED-001":lambda f:f[f.experiment_id=="SYN-BALANCED-001"],"Anomaly images only":lambda f:f[f.image_outcome=="anomaly_present"],"Clean/no-anomaly images only":lambda f:f[f.image_outcome=="no_anomaly"],"Thin cracks only":lambda f:f[f.anomaly_type=="thin_crack"],"Pitting clusters only":lambda f:f[f.anomaly_type=="pitting_cluster"],"Specular highlights only":lambda f:f[f.anomaly_type=="specular_highlights"],"Successful runs only":lambda f:f[f.run_status=="completed"],"Failed runs only":lambda f:f[f.run_status=="failed"],"True anomaly found":lambda f:f[f.first_true_anomaly_proposal_rank.notna()],"True anomaly missed":lambda f:f[(f.image_outcome=="anomaly_present")&f.first_true_anomaly_proposal_rank.isna()],"False-positive cases":lambda f:f[f.false_positive_proposals>0],"Zero-proposal clean images":lambda f:f[(f.image_outcome=="no_anomaly")&(f.final_proposals==0)],"Refined contextual only":lambda f:f[f.method=="refined contextual method"],"Multi-scale fused only":lambda f:f[f.method=="multi-scale fused method"],"Compare advanced methods":lambda f:f[f.method.isin(["multi-scale fused method","refined contextual method"])],"Compare all methods for selected image":lambda f:f[f.image_filename==selected_image],
    }
    if quick in quick_map: result=quick_map[quick](result)
    order=[column for column in (sort_column,secondary) if column and column in result]
    return result.sort_values(order,ascending=ascending,na_position="last") if order else result


def select_analysis_scope(frame,experiment_id,experiment_version,dataset_id=None,dataset_version=None,dataset_split=None):
    result=frame[(frame.experiment_id==experiment_id)&(frame.experiment_version.astype(int)==int(experiment_version))]
    for column,value in (("dataset_id",dataset_id),("dataset_version",dataset_version),("dataset_split",dataset_split)):
        if value not in (None,""):result=result[result[column]==value]
    return result.copy()


def scope_audit(frame):
    duplicate_keys=[*PAIR_KEYS,"method"]; duplicates=int(frame.duplicated(duplicate_keys,keep=False).sum()) if not frame.empty else 0; advanced=frame[frame.method.isin(ADVANCED_METHODS)]; fused=set(advanced[advanced.method==ADVANCED_METHODS[0]].image_id); contextual=set(advanced[advanced.method==ADVANCED_METHODS[1]].image_id)
    images=frame.drop_duplicates("image_id")
    return {"selected_experiment_id":frame.experiment_id.iloc[0] if frame.experiment_id.nunique()==1 else "multiple","selected_experiment_version":int(frame.experiment_version.iloc[0]) if frame.experiment_version.nunique()==1 else "multiple","source_result_rows":len(frame),"unique_images":frame.image_id.nunique(),"unique_methods":frame.method.nunique(),"rows_per_method":frame.method.value_counts().to_dict(),"anomaly_present_images":int((images.image_outcome=="anomaly_present").sum()),"clean_no_anomaly_images":int((images.image_outcome=="no_anomaly").sum()),"category_counts":images.anomaly_type.value_counts().to_dict(),"duplicate_image_method_rows":duplicates,"fused_rows":int((advanced.method==ADVANCED_METHODS[0]).sum()),"contextual_rows":int((advanced.method==ADVANCED_METHODS[1]).sum()),"paired_image_count":len(fused&contextual),"fused_only_image_count":len(fused-contextual),"contextual_only_image_count":len(contextual-fused),"unmatched_advanced_method_rows":len(fused^contextual)}


def category_summary(frame:pd.DataFrame)->pd.DataFrame:
    completed=frame[frame.run_status=="completed"].copy(); rows=[]
    for (category,method),group in completed.groupby(["anomaly_type","method"],dropna=False):
        positive=group[group.image_outcome=="anomaly_present"]; clean=group[group.image_outcome=="no_anomaly"]
        ranks=positive.first_true_anomaly_proposal_rank.dropna()
        row={"category":category or "unclassified","method":method,"images":group.image_id.nunique(),"anomaly_present_images":positive.image_id.nunique(),"clean_images":clean.image_id.nunique(),"proposal_precision":group.proposal_precision.mean(),"proposal_precision_std":group.proposal_precision.std(),"proposal_recall":positive.proposal_recall.mean() if len(positive) else np.nan,"proposal_recall_std":positive.proposal_recall.std() if len(positive)>1 else np.nan,"mean_iou":positive.mean_iou.mean() if len(positive) else np.nan,"best_iou":positive.best_iou.mean() if len(positive) else np.nan,"mean_false_proposals_per_image":group.false_positive_proposals.mean(),"false_negative_count":positive.false_negative_anomalies.sum(),"mean_first_true_anomaly_rank":ranks.mean() if len(ranks) else np.nan,"median_first_true_anomaly_rank":ranks.median() if len(ranks) else np.nan,"mean_processing_time":group.processing_time_seconds.mean(),"processing_time_std":group.processing_time_seconds.std(),"correct_zero_proposal_clean_images":int(((clean.final_proposals==0)).sum())}
        for k in TOP_K: row[f"top_{k}_proposal_recall"]=positive[f"top_{k}_hit"].mean() if len(positive) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def positive_negative_summary(frame):
    data=frame[frame.run_status=="completed"].copy(); rows=[]
    for (outcome,method),group in data.groupby(["image_outcome","method"]):
        positive=group[group.image_outcome=="anomaly_present"]
        rows.append({"image_outcome":outcome,"method":method,"images":group.image_id.nunique(),"precision":group.proposal_precision.mean(),"recall":positive.proposal_recall.mean() if len(positive) else np.nan,"false_proposals":group.false_positive_proposals.mean(),"zero_proposal_images":int((group.final_proposals==0).sum())})
    return pd.DataFrame(rows)


def paired_advanced(frame:pd.DataFrame)->pd.DataFrame:
    duplicate_keys=[*PAIR_KEYS,"method"]
    if frame.duplicated(duplicate_keys,keep=False).any():raise PairingIntegrityError("Duplicate experiment/version/image/method rows prevent strict paired analysis.")
    methods=list(ADVANCED_METHODS); data=frame[(frame.method.isin(methods))&(frame.run_status=="completed")].copy()
    if data.duplicated(duplicate_keys,keep=False).any():raise PairingIntegrityError("Duplicate experiment/version/image/method rows prevent strict paired analysis.")
    if data.experiment_id.nunique()!=1 or data.experiment_version.nunique()!=1:raise PairingIntegrityError("Strict pairing requires one experiment ID and one experiment version.")
    for column in ("dataset_id","dataset_version","dataset_split"):
        if column in data and data[column].nunique()!=1:raise PairingIntegrityError(f"Strict pairing requires one {column}.")
    fused=set(data[data.method==methods[0]].image_id); contextual=set(data[data.method==methods[1]].image_id)
    if fused!=contextual:raise PairingIntegrityError(f"Advanced-method image mismatch: fused-only={len(fused-contextual)}, contextual-only={len(contextual-fused)}")
    index=list(PAIR_KEYS); metadata_columns=["image_filename","anomaly_type","image_outcome"]
    for column in metadata_columns:
        if data.groupby(index)[column].nunique(dropna=False).gt(1).any():raise PairingIntegrityError(f"Paired rows disagree on {column} for the same image ID.")
    metadata=data.drop_duplicates(index).set_index(index)[metadata_columns]
    metrics=["final_proposals","top_1_hit","top_3_hit","top_5_hit","top_8_hit","proposal_precision","proposal_recall","mean_iou","best_iou","false_positive_proposals","processing_time_seconds","first_true_anomaly_proposal_rank"]
    pivot=data.pivot(index=index,columns="method",values=metrics); rows=[]
    for key,values in pivot.iterrows():
        row=dict(zip(index,key if isinstance(key,tuple) else (key,)))
        for column in metadata_columns:row[column]=metadata.loc[key,column]
        for metric in metrics:
            fused=values.get((metric,methods[0]),np.nan); contextual=values.get((metric,methods[1]),np.nan); row[f"fused_{metric}"]=fused; row[f"contextual_{metric}"]=contextual; row[f"difference_{metric}"]=contextual-fused if pd.notna(fused) and pd.notna(contextual) else np.nan
        for metric in ("proposal_precision","proposal_recall","mean_iou","best_iou","false_positive_proposals","processing_time_seconds","first_true_anomaly_proposal_rank"):
            row[f"{metric}_outcome"]=_metric_outcome(row.get(f"fused_{metric}"),row.get(f"contextual_{metric}"),lower_better=metric in {"false_positive_proposals","processing_time_seconds","first_true_anomaly_proposal_rank"})
        row["outcome_label"],row["outcome_reason"]=_paired_label(row); rows.append(row)
    return pd.DataFrame(rows)


def _metric_outcome(fused,contextual,lower_better=False,epsilon=1e-9):
    if pd.isna(fused) or pd.isna(contextual):return "incomparable"
    difference=float(contextual)-float(fused)
    if abs(difference)<=epsilon:return "equal"
    contextual_better=difference<0 if lower_better else difference>0
    return "contextual better" if contextual_better else "fused better"


def _paired_label(row):
    detection=[row["proposal_precision_outcome"],row["proposal_recall_outcome"]] if row["image_outcome"]=="anomaly_present" else []
    false_alarm=row["false_positive_proposals_outcome"]; localization=[row["mean_iou_outcome"],row["best_iou_outcome"]] if row["image_outcome"]=="anomaly_present" else []
    primary=detection+[false_alarm]; comparable=[value for value in primary if value!="incomparable"]
    if not comparable:return "incomparable because metric is undefined","No comparable detection or false-alarm metric."
    contextual=any(value=="contextual better" for value in comparable); fused=any(value=="fused better" for value in comparable)
    if contextual and not fused:return "contextual better","Detection and/or false-alarm metric improved without a primary-metric loss."
    if fused and not contextual:return "fused better","Detection and/or false-alarm metric favoured fused without a primary-metric loss."
    if contextual and fused:return "incomparable because metric is undefined","Primary metrics disagree; no global superiority claim."
    local_context=any(value=="contextual better" for value in localization); local_fused=any(value=="fused better" for value in localization)
    reason="Primary detection and false-alarm metrics are equal. "
    if local_context and not local_fused:reason+="Contextual localisation IoU is higher, reported separately."
    elif local_fused and not local_context:reason+="Fused localisation IoU is higher, reported separately."
    else:reason+="Localisation is also equal or undefined."
    return "equal",reason


def win_tie_loss(paired,category=None,outcome=None):
    data=paired
    if category:data=data[data.anomaly_type==category]
    if outcome:data=data[data.image_outcome==outcome]
    counts=data.outcome_label.value_counts(); return {"contextual_wins":int(counts.get("contextual better",0)),"fused_wins":int(counts.get("fused better",0)),"ties":int(counts.get("equal",0)),"incomparable":int(counts.get("incomparable because metric is undefined",0)),"images":len(data)}


def interpretation(paired):
    clean=paired[paired.image_outcome=="no_anomaly"]; positive=paired[paired.image_outcome=="anomaly_present"]
    suppressed=int((clean.difference_false_positive_proposals<0).sum()); localized=int((positive.difference_mean_iou>0).sum()); ties=int((paired.outcome_label=="equal").sum())
    detection_changed=int(((positive.proposal_precision_outcome!="equal")|(positive.proposal_recall_outcome!="equal")).sum())
    if not suppressed and not localized:return f"No measurable improvement was observed under the current benchmark; primary outcomes were equal on {ties} images."
    return f"Contextual refinement improved localisation IoU on {localized} of {len(positive)} eligible anomaly images and false-alarm suppression on {suppressed} of {len(clean)} clean images. Precision or recall differed on {detection_changed} anomaly images; localisation alone is not treated as global superiority."


def bootstrap_ci(values,samples=1000,seed=42,confidence=.95):
    values=np.asarray(pd.Series(values).dropna(),dtype=float)
    if len(values)<2:return (np.nan,np.nan)
    rng=np.random.default_rng(seed); estimates=np.mean(rng.choice(values,(int(samples),len(values)),replace=True),axis=1); alpha=(1-confidence)/2; return tuple(np.quantile(estimates,[alpha,1-alpha]))


def paired_bootstrap(paired,metric,samples=1000,seed=42):
    column=f"difference_{metric}"; return bootstrap_ci(paired[column],samples,seed) if column in paired else (np.nan,np.nan)


def bootstrap_input_audit(paired,metric):
    column=f"difference_{metric}"; eligible=paired
    if metric in {"proposal_recall","mean_iou","best_iou"}:eligible=eligible[eligible.image_outcome=="anomaly_present"]
    values=eligible[column].dropna() if column in eligible else pd.Series(dtype=float); return {"eligible_paired_images":len(values),"excluded_paired_images":len(paired)-len(values),"exclusion_reason":"Undefined metric or ineligible clean image" if len(values)<len(paired) else "None","paired_keys":eligible.loc[values.index,list(PAIR_KEYS)].to_dict("records") if len(values) else []}


def failure_galleries(paired):
    result={key:[] for key in ("contextual localisation better","fused localisation better","detection disagreement","false-positive disagreement","equal results","incomparable/undefined")}
    for _,row in paired.iterrows():
        if row.outcome_label.startswith("incomparable"):group="incomparable/undefined"
        elif row.proposal_precision_outcome not in {"equal","incomparable"} or row.proposal_recall_outcome not in {"equal","incomparable"}:group="detection disagreement"
        elif row.false_positive_proposals_outcome not in {"equal","incomparable"}:group="false-positive disagreement"
        elif row.mean_iou_outcome=="contextual better":group="contextual localisation better"
        elif row.mean_iou_outcome=="fused better":group="fused localisation better"
        else:group="equal results"
        result[group].append(row.to_dict())
    return {key:pd.DataFrame(value) for key,value in result.items()}


def filtered_csv(frame):return frame.to_csv(index=False).encode()
def filtered_json(frame):return frame.to_json(orient="records",indent=2).encode()

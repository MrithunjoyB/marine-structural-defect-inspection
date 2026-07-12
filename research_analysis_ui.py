"""Streamlit research-result browser and analysis views."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from ablation_study import ABLATION_CONFIGS,CONFIG_BY_ID,ablation_leaderboard,contribution_table,execute_ablation_plan,save_ablation_plan
from registered_experiment import load_ground_truth,load_plan,selected_images
from research_analysis import bootstrap_ci,category_summary,enrich_results,filter_results,filtered_csv,filtered_json,interpretation,paired_advanced,paired_bootstrap,positive_negative_summary,win_tie_loss


ESSENTIAL=["experiment_id","experiment_version","image_filename","anomaly_type","image_outcome","method","run_status"]
METRICS=["final_proposals","first_true_anomaly_proposal_rank","top_1_hit","top_3_hit","top_5_hit","top_8_hit","proposal_precision","proposal_recall","mean_iou","best_iou","false_positive_proposals","false_negative_anomalies","processing_time_seconds"]
PROVENANCE=["plan_id","dataset_id","dataset_version","dataset_split","image_id","review_status","recorded_timestamp"]
FILTER_KEYS=[]


def render_research_analysis(store,registry):
    raw=store.dataframe(); frame=enrich_results(raw,registry)
    if frame.empty:return
    st.markdown("#### Automatic Result Browser")
    search=st.text_input("Search result rows",key="result_browser_search",placeholder="Experiment, image, category, method, status...")
    quick_options=["","Current selected experiment","SYN-BALANCED-001","Anomaly images only","Clean/no-anomaly images only","Thin cracks only","Pitting clusters only","Specular highlights only","Successful runs only","Failed runs only","True anomaly found","True anomaly missed","False-positive cases","Zero-proposal clean images","Refined contextual only","Multi-scale fused only","Compare advanced methods","Compare all methods for selected image"]
    quick=st.selectbox("Quick filter",quick_options,key="result_browser_quick")
    selected_experiment=st.selectbox("Current selected experiment",[""]+sorted(frame.experiment_id.unique()),key="result_browser_selected_experiment")
    selected_image=st.selectbox("Image-centric comparison",[""]+sorted(frame.image_filename.unique()),key="result_browser_selected_image")
    mapping={"experiment_id":"Experiment ID","experiment_version":"Experiment version","dataset_id":"Dataset ID","dataset_version":"Dataset version","dataset_split":"Dataset split","image_filename":"Image filename","anomaly_type":"Anomaly type","clean_artefact_type":"Clean artefact type","image_outcome":"Image outcome","method":"Proposal method","review_status":"Review status","run_status":"Run status"}; filters={}
    with st.expander("Result filters",expanded=True):
        columns=st.columns(3)
        for index,(field,label) in enumerate(mapping.items()):
            key=f"result_filter_{field}"; FILTER_KEYS.append(key); filters[field]=columns[index%3].multiselect(label,sorted(frame[field].dropna().unique().tolist()),key=key)
    numeric={}
    with st.expander("Metric filters"):
        ranges={"final_proposals":"Final proposals","first_true_anomaly_proposal_rank":"First true-anomaly proposal rank","processing_time_seconds":"Processing time","false_positive_proposals":"False-positive proposals","false_negative_anomalies":"False-negative anomalies"}
        for field,label in ranges.items():
            values=pd.to_numeric(frame[field],errors="coerce").dropna(); low=float(values.min()) if len(values) else 0.; actual_high=float(values.max()) if len(values) else 0.; high=actual_high if actual_high>low else low+1.; selected=st.slider(label,low,high,(low,actual_high),key=f"result_range_{field}")
            if selected!=(low,actual_high): numeric[field]=selected
        for field,label in (("proposal_precision","Minimum proposal precision"),("proposal_recall","Minimum proposal recall"),("mean_iou","Minimum mean IoU"),("best_iou","Minimum best IoU")):
            minimum=st.slider(label,0.,1.,0.,.01,key=f"result_min_{field}")
            if minimum>0:numeric[field]=(minimum,None)
        hit_columns=st.columns(4)
        for index,k in enumerate((1,3,5,8)):
            choice=hit_columns[index].selectbox(f"Top-{k} hit",["Any","Hit","Miss","N/A"],key=f"result_top_{k}")
            if choice!="Any": filters[f"top_{k}_hit"]={"Hit":[1,True],"Miss":[0,False],"N/A":[None]}[choice]
    sort_columns=list(frame.columns); sort=st.selectbox("Sort column",sort_columns,index=sort_columns.index("recorded_timestamp"),key="result_sort"); secondary=st.selectbox("Secondary sort",[""]+sort_columns,key="result_secondary_sort"); ascending=st.radio("Sort direction",["Descending","Ascending"],horizontal=True,key="result_sort_direction")=="Ascending"
    if st.button("Reset all filters",on_click=_reset_filters): st.rerun()
    filtered=filter_results(frame,search,filters,numeric,quick,selected_experiment,selected_image,sort,ascending,secondary)
    st.markdown(f"**Showing {len(filtered)} of {len(frame)} result rows**")
    counters=st.columns(3); counters[0].metric("Selected experiments",filtered.experiment_id.nunique()); counters[1].metric("Selected images",filtered.image_id.nunique()); counters[2].metric("Selected methods",filtered.method.nunique())
    if filtered.empty: st.info("No result rows match the current filters."); return
    view=st.radio("Visible columns",["Essential columns","Metrics","Provenance","All columns","Custom visible columns"],horizontal=True,key="result_column_view")
    if view=="Essential columns":visible=ESSENTIAL
    elif view=="Metrics":visible=ESSENTIAL[:3]+METRICS
    elif view=="Provenance":visible=ESSENTIAL[:3]+PROVENANCE
    elif view=="Custom visible columns":visible=st.multiselect("Custom visible columns",list(filtered.columns),default=ESSENTIAL,key="result_custom_columns")
    else:visible=list(filtered.columns)
    st.dataframe(filtered[[column for column in visible if column in filtered]],width="stretch",hide_index=True)
    exports=st.columns(2); exports[0].download_button("Download filtered CSV",filtered_csv(filtered),"filtered_results.csv","text/csv"); exports[1].download_button("Download filtered JSON",filtered_json(filtered),"filtered_results.json","application/json")
    if selected_image:
        candidates=frame[frame.image_filename==selected_image]; experiments=sorted(candidates.experiment_id.unique()); default=experiments.index(selected_experiment) if selected_experiment in experiments else 0; comparison_experiment=st.selectbox("Image comparison experiment",experiments,index=default,key="result_image_comparison_experiment"); comparison=candidates[candidates.experiment_id==comparison_experiment]; st.markdown("##### Selected-image Method Comparison"); st.dataframe(comparison[[c for c in ESSENTIAL+METRICS if c in comparison]],width="stretch",hide_index=True)
    _result_details(filtered,registry)
    _category_evaluation(filtered)
    _paired_evaluation(filtered)
    _statistical_reporting(filtered)
    _ablation_study(store,registry,frame)


def _reset_filters():
    prefixes=("result_browser_","result_filter_","result_range_","result_min_","result_top_","result_sort","result_secondary_","result_column_")
    for key in list(st.session_state):
        if key.startswith(prefixes): del st.session_state[key]


def _result_details(filtered,registry):
    st.markdown("##### Result Details")
    row_id=st.selectbox("Selected result row",filtered.result_id.tolist(),format_func=lambda value:f"{filtered.loc[filtered.result_id==value,'image_filename'].iloc[0]} | {filtered.loc[filtered.result_id==value,'method'].iloc[0]}",key="result_detail_row")
    with st.expander("Expand selected result details"):
        row=filtered[filtered.result_id==row_id].iloc[0]; image=registry.images(row.dataset_id); image=image[image.image_id==row.image_id].iloc[0]; columns=st.columns(3); columns[0].image(str(registry.root/"raw"/row.dataset_id/image.stored_filename),caption="Original image",width="stretch"); columns[1].image(load_ground_truth(image),caption="Ground-truth mask",width="stretch"); columns[2].image(row.visualization_path,caption="Matched and unmatched proposals",width="stretch")
        st.json({"method":row.method,"anomaly_category":row.anomaly_type,"precision":row.proposal_precision,"recall":None if pd.isna(row.proposal_recall) else row.proposal_recall,"false_positives":row.false_positive_proposals,"false_negatives":row.false_negative_anomalies,"processing_time":row.processing_time_seconds,"experiment":f"{row.experiment_id} v{row.experiment_version}","dataset":f"{row.dataset_id} {row.dataset_version} {row.dataset_split}","proposal_ranks_and_iou":json.loads(row.proposal_details_json),"execution_configuration":json.loads(row.configuration_json) if row.configuration_json else {}})


def _category_evaluation(frame):
    st.markdown("### Category-wise Evaluation")
    summary=category_summary(frame); positive=summary[summary.anomaly_present_images>0]; clean=summary[summary.clean_images>0]
    st.caption("Recall denominators include anomaly-present eligible images only. Undefined clean-only recall is displayed as N/A.")
    st.markdown("##### Anomaly-category comparison"); st.dataframe(_na_table(positive),width="stretch",hide_index=True)
    st.markdown("##### Clean-artefact robustness comparison"); st.dataframe(_na_table(clean),width="stretch",hide_index=True)
    st.markdown("##### Positive versus negative performance"); st.dataframe(_na_table(positive_negative_summary(frame)),width="stretch",hide_index=True)
    st.markdown("##### Method-by-category matrix"); st.dataframe(_na_table(summary.pivot(index="category",columns="method",values="proposal_precision")),width="stretch")
    charts=((positive,"top_1_proposal_recall","Top-1 recall by anomaly category"),(positive,"proposal_precision","Precision by anomaly category"),(positive,"proposal_recall","Recall by anomaly category"),(clean,"mean_false_proposals_per_image","False proposals by clean artefact category"),(positive,"mean_iou","Mean IoU by anomaly category"),(summary,"mean_processing_time","Processing time by category"),(positive,"mean_first_true_anomaly_rank","First true-anomaly rank distribution"))
    for data,column,title in charts:
        valid=data[["category","method",column]].dropna() if not data.empty else data
        if valid.empty: st.info(f"{title}: N/A for the current filtered rows.")
        else: st.markdown(f"##### {title}"); st.bar_chart(valid,x="category",y=column,color="method",width="stretch")


def _paired_evaluation(frame):
    st.markdown("### Multi-scale Fused vs Refined Contextual")
    paired=paired_advanced(frame)
    if paired.empty: st.info("No identical-image advanced-method pairs are available."); return
    st.dataframe(paired,width="stretch",hide_index=True); st.json({"complete_balanced_test":win_tie_loss(paired),"thin_cracks":win_tie_loss(paired,"thin_crack"),"pitting_clusters":win_tie_loss(paired,"pitting_cluster"),"clean_normal_texture":win_tie_loss(paired,"normal_texture"),"specular_highlights":win_tie_loss(paired,"specular_highlights"),"all_anomaly_present":win_tie_loss(paired,outcome="anomaly_present"),"all_clean":win_tie_loss(paired,outcome="no_anomaly")}); st.info(interpretation(paired))
    difference_columns=[column for column in paired if column.startswith("difference_") and any(key in column for key in ("precision","recall","iou","false_positive","processing_time","first_true"))]
    for column in difference_columns:
        valid=paired[["image_filename",column]].dropna()
        if not valid.empty: st.bar_chart(valid,x="image_filename",y=column,width="stretch")
    st.markdown("##### Failure-case galleries"); gallery=paired.assign(case=np.select([paired.difference_false_positive_proposals.lt(0),paired.difference_false_positive_proposals.gt(0),paired.outcome_label.eq("equal")],["contextual beats fused","fused beats contextual","ties"],default="other")); st.dataframe(gallery[["image_filename","anomaly_type","image_outcome","case","outcome_label"]],width="stretch",hide_index=True)


def _statistical_reporting(frame):
    st.warning("Results are preliminary and based on a controlled synthetic benchmark. They do not establish real-world marine inspection performance.")
    samples=int(st.number_input("Bootstrap samples",100,10000,1000,100,key="bootstrap_samples")); seed=int(st.number_input("Bootstrap seed",0,1000000,42,key="bootstrap_seed")); paired=paired_advanced(frame)
    rows=[]
    for metric in ("proposal_precision","proposal_recall","false_positive_proposals","mean_iou"):
        low,high=paired_bootstrap(paired,metric,samples,seed); rows.append({"paired_difference_metric":metric,"images":int(paired[f"difference_{metric}"].notna().sum()) if f"difference_{metric}" in paired else 0,"ci_low":low,"ci_high":high})
    st.markdown("##### Paired bootstrap confidence intervals"); st.dataframe(_na_table(pd.DataFrame(rows)),width="stretch",hide_index=True)
    if len(paired)<10: st.info("Sample size is too small for significance claims; intervals are descriptive only.")


def _ablation_study(store,registry,frame):
    st.markdown("### Ablation Study")
    with registry.connect() as con: plans=[dict(row) for row in con.execute("SELECT plan_id,experiment_id FROM experiment_plans ORDER BY created_timestamp DESC")]
    plan_id=st.selectbox("Ablation source experiment",[row["plan_id"] for row in plans],format_func=lambda value:next(row["experiment_id"] for row in plans if row["plan_id"]==value),key="ablation_source_plan"); plan=load_plan(registry,plan_id)
    selected_ids=st.multiselect("Selected ablation configurations",[item.configuration_id for item in ABLATION_CONFIGS],default=["ABL-FULL"],format_func=lambda value:CONFIG_BY_ID[value].name,key="ablation_configurations"); experiment_id=st.text_input("Ablation experiment ID",value=f"ABL-{plan['experiment_id']}"); version=int(st.number_input("Ablation experiment version",1,1000,1)); reviewer=st.text_input("Ablation reviewer"); seed=int(st.number_input("Ablation seed",0,1000000,int(plan["configuration"].get("random_seed",42)))); criteria=st.columns(2); iou=float(criteria[0].number_input("Ablation IoU threshold",0.,1.,.1,.01)); overlap=float(criteria[1].number_input("Ablation overlap threshold",0.,1.,.25,.01))
    manifest_hash=plan["manifest_hash"]; definitions=[CONFIG_BY_ID[value] for value in selected_ids]; path=registry.root/"exports"/f"{experiment_id}_ablation_plan.json"
    if st.button("Preview Ablation Plan"):
        payload=save_ablation_plan(path,plan,definitions,experiment_id,version,reviewer,seed,{"iou":iou,"mask_overlap":overlap},manifest_hash); st.json(payload)
    if st.button("Execute Ablation Study",disabled=not definitions):
        save_ablation_plan(path,plan,definitions,experiment_id,version,reviewer,seed,{"iou":iou,"mask_overlap":overlap},manifest_hash); results=execute_ablation_plan(registry,store,plan_id,definitions,version,iou,overlap,experiment_id=experiment_id); st.success(f"Stored {len(results)} ablation result rows."); st.rerun()
    all_results=store.dataframe(); ablation=all_results[all_results.method.isin([item.configuration_id for item in ABLATION_CONFIGS])]
    if not ablation.empty:
        leaderboard=ablation_leaderboard(ablation); st.markdown("##### Ablation Leaderboard"); st.dataframe(leaderboard,width="stretch",hide_index=True); st.markdown("##### Ablation Contribution Table"); st.caption("Empirical benchmark differences, not causal proof."); st.dataframe(contribution_table(leaderboard),width="stretch",hide_index=True)
        st.download_button("Download ablation results CSV",ablation.to_csv(index=False).encode(),"ablation_results.csv","text/csv"); st.download_button("Download ablation results JSON",ablation.to_json(orient="records",indent=2).encode(),"ablation_results.json","application/json");
    if path.exists(): st.download_button("Download ablation configuration JSON",path.read_bytes(),path.name,"application/json")


def _na_table(frame):
    return frame.apply(lambda column:column.map(lambda value:"N/A" if pd.isna(value) else str(value)))

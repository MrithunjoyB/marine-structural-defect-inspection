"""Streamlit renderer for persistent Research Evaluation workflows."""

from __future__ import annotations

from datetime import date, datetime
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import streamlit as st

from experiment_tracking import (
    EXPERIMENT_STATUSES,
    GROUND_TRUTH_STATUSES,
    IMAGE_TABLE_COLUMNS,
    METHOD_NAMES,
    DuplicateRecordError,
    ExperimentStore,
    build_experiment_records,
    delete_legacy_records,
    experiment_tables,
    legacy_record_indices,
    load_legacy_records,
    migrate_legacy_records,
    records_to_csv,
    records_to_json,
    with_version,
)
from research_dataset import DatasetRegistry, FINAL_STATUS
from registered_experiment import RegisteredExperimentStore, execute_plan, load_ground_truth, load_plan, method_summary, selected_images
from research_analysis_ui import render_research_analysis


RECALL_CHART_COLUMNS = {
    "Top-1": "top_1_proposal_recall",
    "Top-3": "top_3_proposal_recall",
    "Top-5": "top_5_proposal_recall",
    "Top-8": "top_8_proposal_recall",
}
OUTCOME_CHART_COLUMNS = {
    "Accepted": "mean_accepted_proposals_per_image",
    "Rejected": "mean_rejected_proposals_per_image",
    "Uncertain": "mean_uncertain_proposals_per_image",
    "Not reviewed": "mean_not_reviewed_proposals_per_image",
}
EFFICIENCY_CHART_COLUMNS = {
    "Annotation acceptance rate": "annotation_acceptance_rate",
    "Mean review time (seconds)": "mean_review_time_seconds",
    "Mean proposals before first useful": "mean_proposals_reviewed_before_first_useful",
}
FALSE_PROPOSAL_COLUMN = "mean_false_proposals_per_image"
METHOD_DISPLAY_NAMES={
    "contour-only baseline":"Contour baseline",
    "fixed-threshold baseline":"Fixed-threshold baseline",
    "multi-scale fused method":"Multi-scale fused",
    "refined contextual method":"Refined contextual",
}
EMPTY_METRIC_MESSAGES={
    "annotation_acceptance_rate":"No accepted or rejected proposals are available, so acceptance rate is undefined.",
    "mean_proposals_reviewed_before_first_useful":"No reviewer-confirmed useful proposal rank is available.",
    "mean_false_proposals_per_image":"No manually reviewed rejected proposals are available for this metric.",
}


def render_research_evaluation(
    output_dir: Path,
    image_name: str | None,
    annotations,
    proposal_result,
    feature_maps,
    review_start_time: str | None,
    review_completion_time: str | None,
    preprocessing_settings: dict | None = None,
    path_resolver=None,
) -> None:
    st.subheader("Research Evaluation")
    database_path = Path(os.environ.get("STRUCTVISION_RESEARCH_DB", output_dir / "research_evaluation.sqlite3"))
    legacy_json_path = Path(os.environ.get("STRUCTVISION_LEGACY_EXPERIMENT_JSON", output_dir / "research_experiment_results.json"))
    store = ExperimentStore(database_path)
    _render_recording(
        store, image_name, annotations, proposal_result, feature_maps,
        review_start_time, review_completion_time,
    )
    automatic_store=RegisteredExperimentStore(output_dir / "registered_experiment_results.sqlite3"); registry=DatasetRegistry(output_dir.parent/"research_data")
    _render_registered_dataset_experiment(output_dir.parent, preprocessing_settings or {},automatic_store,path_resolver)
    st.divider()
    _render_dashboard_and_management(store,automatic_store,registry,path_resolver)
    st.divider()
    _render_legacy_migration(store, legacy_json_path)


def _render_registered_dataset_experiment(base_dir:Path,preprocessing_settings:dict,automatic_store,path_resolver=None)->None:
    st.markdown("### Create Experiment from Registered Dataset")
    registry=DatasetRegistry(base_dir/"research_data"); datasets=registry.datasets()
    if datasets.empty:
        st.info("Register a dataset in Research Dataset Intake before creating a reproducible experiment."); return
    dataset_id=st.selectbox("Registered dataset",datasets.dataset_id.unique(),key="registered_experiment_dataset")
    versions=datasets[datasets.dataset_id==dataset_id].dataset_version.tolist(); version=st.selectbox("Registered dataset version",versions)
    images=registry.images(dataset_id); splits=["all"]+sorted(value for value in images.split.unique().tolist() if value and value!="unassigned")
    row=st.columns(3); split=row[0].selectbox("Dataset split",splits); subset=row[1].number_input("Subset size",1,max(len(images),1),min(len(images),100) if len(images) else 1); seed=row[2].number_input("Experiment random seed",0,1000000,42)
    subset_filter=st.selectbox("Experiment subset filter",["all","anomaly-present only","no-anomaly only","balanced positive/negative subset"])
    anomaly_options=sorted(images[images.image_outcome=="anomaly_present"].anomaly_type.dropna().unique().tolist()); clean_options=sorted(images[images.image_outcome=="no_anomaly"].anomaly_type.dropna().unique().tolist())
    filter_columns=st.columns(2); selected_anomalies=filter_columns[0].multiselect("Selected anomaly types",anomaly_options); selected_clean=filter_columns[1].multiselect("Selected clean artefact types",clean_options)
    row=st.columns(3); status=row[0].selectbox("Registered experiment status",["Development / Test",FINAL_STATUS]); reviewer=row[1].text_input("Registered experiment reviewer"); experiment_id=row[2].text_input("Registered experiment ID")
    methods=st.multiselect("Proposal methods",list(METHOD_NAMES),default=list(METHOD_NAMES))
    default_config={"preprocessing":preprocessing_settings,"proposal":{},"feature_weights":{},"thresholds":{},"border_margin":preprocessing_settings.get("border_margin"),"maximum_regions":preprocessing_settings.get("max_regions"),"ablation":{key:value for key,value in preprocessing_settings.items() if key.startswith("use_")}}
    configuration_text=st.text_area("Parameter configuration (JSON)",value=json.dumps(default_config,indent=2),height=220)
    override=st.checkbox("Override final-experiment provenance/licence restrictions",help="Use only after reviewing the warning and documenting the reason.")
    if st.button("Create Registered Dataset Experiment"):
        try:
            if not experiment_id.strip() or not reviewer.strip() or not methods: raise ValueError("Experiment ID, reviewer, and at least one method are required")
            parameters=json.loads(configuration_text)
            path=registry.create_experiment_plan(experiment_id,dataset_id,version,split,int(subset),status,reviewer,methods,parameters,int(seed),override,subset_filter,selected_anomalies,selected_clean)
            st.success("Created reproducible registered-dataset experiment plan.")
            st.download_button("Download Experiment Configuration JSON",path.read_bytes(),path.name,"application/json")
        except (ValueError,json.JSONDecodeError) as error: st.error(str(error))
    _render_registered_execution(registry,automatic_store)


def _render_registered_execution(registry,store):
    st.markdown("### Execute Registered Dataset Experiment")
    with registry.connect() as con: plans=[dict(row) for row in con.execute("SELECT * FROM experiment_plans ORDER BY created_timestamp DESC").fetchall()]
    if not plans: st.info("Create a registered-dataset experiment plan before execution."); return
    labels={row["plan_id"]:f"{row['experiment_id']} | {row['dataset_id']} {row['dataset_version']} | {row['split']}" for row in plans}
    plan_id=st.selectbox("Saved experiment plan",list(labels),format_func=labels.get,key="execution_plan")
    plan=load_plan(registry,plan_id); version=int(st.number_input("Execution version",1,10000,1,key="registered_execution_version")); state=store.execution(plan_id,version)
    st.json({"execution_status":state["status"],"selected_images":len(plan["selected_image_ids"]),"methods":plan["configuration"].get("proposal_methods",[]),"completed_pairs":state.get("completed_pairs",0),"total_pairs":state.get("total_pairs",0)})
    criteria=st.columns(2); iou=float(criteria[0].slider("IoU threshold",0.,1.,.10,.01)); overlap=float(criteria[1].slider("Mask overlap threshold",0.,1.,.25,.01))
    handling=st.radio("Completed-pair handling",["resume","overwrite","create new experiment version"],horizontal=True)
    progress_bar=st.progress(0.); progress_text=st.empty()
    def update(payload):
        progress_bar.progress(min(payload["completed"]/max(payload["total"],1),1.)); progress_text.caption(f"Current image: {payload['current_image']} | Current method: {payload['current_method']} | {payload['completed']}/{payload['total']} runs | Elapsed: {payload['elapsed']:.1f}s | Estimated remaining: {payload['estimated_remaining']:.1f}s")
    controls=st.columns(6)
    execute=controls[0].button("Execute Registered Dataset Experiment",type="primary")
    resume=controls[1].button("Resume Registered Dataset Experiment")
    retry=controls[2].button("Re-run failed pairs")
    if controls[3].button("Cancel execution"): st.session_state.registered_execution_cancel=True; st.warning("Cancellation requested; execution stops after the current pair.")
    create_version=controls[4].button("Create new version")
    delete_confirm=controls[5].checkbox("Confirm delete results")
    if st.button("Delete results",disabled=not delete_confirm): st.success(f"Deleted {store.delete_results(plan_id,version)} automatic rows; the plan was preserved."); st.rerun()
    if execute or resume or retry or create_version:
        try:
            run_version=store.next_version(plan_id) if handling=="create new experiment version" or create_version else version
            mode="retry_failed" if retry else ("resume" if resume or handling=="resume" else "overwrite")
            st.session_state.registered_execution_cancel=False
            results=execute_plan(registry,store,plan_id,run_version,iou,overlap,mode,update,lambda:st.session_state.get("registered_execution_cancel",False))
            st.success(f"Execution finished with {len(results)} persistent image-method result rows."); st.rerun()
        except (ValueError,KeyError) as error: st.error(str(error))
    _render_automatic_results(registry,store,plan_id,version)


def _render_automatic_results(registry,store,plan_id,version):
    results=store.dataframe(plan_id,version)
    if results.empty:return
    st.markdown("#### Registered Dataset Results")
    st.dataframe(results,width="stretch",hide_index=True); summary=method_summary(results); st.markdown("##### Method-level Summary"); st.dataframe(summary,width="stretch",hide_index=True)
    downloads=st.columns(5); downloads[0].download_button("Download result CSV",results.to_csv(index=False).encode(),"registered_results.csv","text/csv"); downloads[1].download_button("Download result JSON",results.to_json(orient="records",indent=2).encode(),"registered_results.json","application/json")
    plan=load_plan(registry,plan_id); selected=selected_images(registry,plan); config=json.dumps(plan,indent=2,default=str).encode(); manifest=selected.to_json(orient="records",indent=2).encode(); report=json.dumps({"execution":store.execution(plan_id,version),"summary":summary.to_dict("records")},indent=2,default=str).encode()
    downloads[2].download_button("Experiment configuration JSON",config,"experiment_configuration.json","application/json"); downloads[3].download_button("Selected image manifest",manifest,"selected_image_manifest.json","application/json"); downloads[4].download_button("Summary report",report,"summary_report.json","application/json")
    if not summary.empty:
        result_images=selected[selected.image_id.isin(results.image_id.unique())]; positive_count=int((result_images.image_outcome=="anomaly_present").sum())
        if positive_count:
            _grouped_chart(summary,{"Top-1":"top_1_proposal_recall","Top-3":"top_3_proposal_recall","Top-5":"top_5_proposal_recall","Top-8":"top_8_proposal_recall"},"Automatic Top-K Recall","Recall",True)
            _grouped_chart(summary,{"Precision":"proposal_precision","Recall":"proposal_recall"},"Automatic Precision / Recall","Score",True)
        else:
            st.markdown("##### False-positive robustness evaluation")
            st.info("Recall is undefined because no positive anomaly images are eligible. Empty recall series are not displayed.")
            _single_chart(summary,"proposal_precision","Automatic Proposal Precision on Clean Images","Precision",percent=True)
        _single_chart(summary,"false_proposals_per_image","Automatic False Proposals per Image","False proposals")
        _single_chart(summary,"processing_time_seconds","Automatic Processing Time","Seconds")
    selected_id=st.selectbox("Selected test image",results.image_id.unique(),format_func=lambda value:results.loc[results.image_id==value,"image_filename"].iloc[0])
    if st.button("Open Selected Test Image"):
        row=selected[selected.image_id==selected_id].iloc[0]; image_path=registry.root/"raw"/row.dataset_id/row.stored_filename; truth=load_ground_truth(row,path_resolver); columns=st.columns(2); columns[0].image(str(image_path),caption="Original image",width="stretch"); columns[1].image(truth,caption="Exact ground-truth mask",width="stretch")
        image_results=results[(results.image_id==selected_id)&(results.run_status=="completed")]
        for _,item in image_results.iterrows(): st.image(item.visualization_path,caption=f"{item.method}: green matched, red unmatched; ranks and IoU shown",width="stretch")
        st.dataframe(image_results[["method","first_true_anomaly_proposal_rank","proposal_details_json"]],width="stretch",hide_index=True)


def _render_recording(store, image_name, annotations, proposal_result, feature_maps, review_start, review_completion):
    st.markdown("### Record Evaluation")
    controls = st.columns(2)
    if controls[0].button("Start New Experiment"):
        for key in list(st.session_state):
            if key.startswith("research_") and key not in {"research_management_message"}:
                del st.session_state[key]
        st.session_state.pending_experiment_records = None
        st.rerun()
    if controls[1].button("Discard Current Unsaved Evaluation"):
        st.session_state.pending_experiment_records = None
        st.info("Discarded the unsaved evaluation preview.")

    identity = st.columns(3)
    experiment_id = identity[0].text_input("Experiment ID", key="research_experiment_id")
    reviewer_id = identity[1].text_input("Reviewer ID", key="research_reviewer_id")
    experiment_version = int(identity[2].number_input("Experiment version", 1, 10000, 1, key="research_experiment_version"))
    status_cols = st.columns(2)
    experiment_status = status_cols[0].selectbox("Experiment status", EXPERIMENT_STATUSES, index=0, key="research_experiment_status")
    image_outcome = status_cols[1].selectbox("Image-level reviewer outcome", ["uncertain", "anomaly present", "no anomaly"], key="research_image_outcome")

    provenance = st.columns(2)
    dataset_source = provenance[0].text_input("Dataset/source name", key="research_dataset_source")
    image_provenance = provenance[1].text_input("Image provenance", key="research_image_provenance")
    safety = st.columns(2)
    license_status = safety[0].text_input("License/status", value="unknown", key="research_license_status")
    ground_truth_status = safety[1].selectbox("Ground-truth availability", GROUND_TRUTH_STATUSES, index=2, key="research_ground_truth_status")
    ground_truth_override = False
    if ground_truth_status == "unknown":
        ground_truth_override = st.checkbox(
            "Override unknown ground truth for recall (use with caution)", value=False,
            help="Includes this record in recall eligibility despite unknown ground truth.",
        )
    development_notes = st.text_area("Development notes", key="research_development_notes")

    if ground_truth_status == "unknown" and experiment_status == "Final Research Evaluation":
        st.warning("Unknown ground truth is excluded from final quantitative recall. Save only if that exclusion is intended.")
    if review_start:
        timing = st.columns(2)
        timing[0].text_input("Review started", review_start, disabled=True)
        timing[1].text_input("Review completed", review_completion or "Not completed", disabled=True)

    if st.button("Preview Method Rows"):
        if proposal_result is None or feature_maps is None or not annotations or not review_completion:
            st.error("Analyze an image and save Human Review metadata before previewing method rows.")
        else:
            try:
                st.session_state.pending_experiment_records = build_experiment_records(
                    experiment_id=experiment_id, reviewer_id=reviewer_id, image_filename=image_name or "unknown",
                    image_outcome=image_outcome, review_start_time=review_start or "",
                    review_completion_time=review_completion, annotations=annotations,
                    proposal_result=proposal_result, feature_maps=feature_maps,
                    experiment_version=experiment_version, experiment_status=experiment_status,
                    dataset_source=dataset_source, image_provenance=image_provenance,
                    license_status=license_status, ground_truth_status=ground_truth_status,
                    ground_truth_recall_override=ground_truth_override,
                    development_notes=development_notes,
                )
            except ValueError as error:
                st.error(str(error))

    pending = st.session_state.get("pending_experiment_records")
    if pending:
        preview = pd.DataFrame([record.to_dict() for record in pending])
        st.caption("Preview: reviewed and not-reviewed method rows")
        st.dataframe(preview[IMAGE_TABLE_COLUMNS], width="stretch")
        duplicates = store.duplicate_count(pending)
        duplicate_action = "cancel"
        if duplicates:
            st.warning(f"{duplicates} duplicate method rows already exist for this experiment version.")
            duplicate_action = st.radio("Duplicate handling", ["cancel", "overwrite", "create new version"], horizontal=True)
        if st.button("Save Experiment Records", type="primary"):
            try:
                records = pending
                action = duplicate_action
                if duplicate_action == "create new version":
                    version = store.next_version(experiment_id)
                    records = with_version(pending, version)
                    action = "cancel"
                saved = store.save(records, duplicate_action=action)
                st.session_state.pending_experiment_records = None
                st.success(f"Saved {saved} method rows.")
            except DuplicateRecordError as error:
                st.error(f"Save cancelled: {error}")


def _render_dashboard_and_management(store: ExperimentStore,automatic_store=None,registry=None,path_resolver=None) -> None:
    st.markdown("### Manage Experiments")
    all_records = store.dataframe()
    automatic=automatic_store.dataframe() if automatic_store else pd.DataFrame()
    if not automatic.empty:
        render_research_analysis(automatic_store,registry,path_resolver)
    if all_records.empty:
        if automatic.empty: st.info("No SQLite experiment records are stored yet.")
        else: st.info("Automatic results are shown above; no separate human-review records are stored.")
        return

    include_development = st.checkbox("Include development records", value=False)
    if include_development:
        scope = st.radio("Dashboard scope", ["Both", "Development only"], horizontal=True)
        development_only = scope == "Development only"
        display_scope = "development only" if development_only else "both final and development"
    else:
        development_only = False
        display_scope = "final only"
    st.info(f"Dashboard currently displays: {display_scope}.")

    filters = _render_filters(all_records)
    filtered = store.dataframe(filters)
    image_table, summary = experiment_tables(
        filtered, include_development=include_development, development_only=development_only,
    )
    if image_table.empty:
        st.warning("No records match the active filters and dashboard scope.")
    else:
        st.caption("Image-level records")
        display_columns = [column for column in IMAGE_TABLE_COLUMNS if column in image_table]
        selection = st.dataframe(
            image_table[display_columns], width="stretch", hide_index=True,
            on_select="rerun", selection_mode="multi-row", key="research_record_selection",
        )
        selected_indices = list(selection.selection.rows)
        selected = image_table.iloc[selected_indices] if selected_indices else image_table.iloc[0:0]
        export_data = selected if not selected.empty else image_table
        export_label = "selected" if not selected.empty else "filtered"
        export_cols = st.columns(2)
        export_cols[0].download_button(f"Download {export_label} records CSV", records_to_csv(export_data), "experiment_records.csv", "text/csv")
        export_cols[1].download_button(f"Download {export_label} records JSON", records_to_json(export_data), "experiment_records.json", "application/json")

        st.caption("Dataset-level summary")
        st.dataframe(_display_na(summary), width="stretch", hide_index=True)
        _render_charts(summary)
        _render_destructive_controls(store, all_records, selected)

    message = st.session_state.pop("research_management_message", None)
    if message:
        st.success(message)


def _render_filters(records: pd.DataFrame) -> dict[str, object]:
    st.caption("Filters")
    columns = st.columns(3)
    experiment_id = columns[0].selectbox("Filter by Experiment ID", [""] + sorted(records["experiment_id"].dropna().unique().tolist()))
    reviewer_id = columns[1].selectbox("Filter by Reviewer ID", [""] + sorted(records["reviewer_id"].dropna().unique().tolist()))
    image_filename = columns[2].selectbox("Filter by image filename", [""] + sorted(records["image_filename"].dropna().unique().tolist()))
    columns = st.columns(2)
    method = columns[0].selectbox("Filter by method", [""] + list(METHOD_NAMES))
    experiment_status = columns[1].selectbox("Filter by experiment status", [""] + list(EXPERIMENT_STATUSES))
    use_dates = st.checkbox("Filter by date range")
    date_from = date_to = None
    if use_dates:
        dates = pd.to_datetime(records["recorded_timestamp"], errors="coerce").dropna()
        minimum = dates.min().date() if not dates.empty else date.today()
        maximum = dates.max().date() if not dates.empty else date.today()
        date_from, date_to = st.date_input("Recorded date range", value=(minimum, maximum))
    return {
        "experiment_id": experiment_id, "reviewer_id": reviewer_id,
        "image_filename": image_filename, "method": method,
        "experiment_status": experiment_status, "date_from": date_from, "date_to": date_to,
    }


def _render_destructive_controls(store: ExperimentStore, all_records: pd.DataFrame, selected: pd.DataFrame) -> None:
    st.caption("Destructive actions require confirmation")
    if not selected.empty:
        confirm = st.checkbox("Confirm deletion of selected rows")
        if st.button("Delete selected rows", disabled=not confirm):
            _delete_and_refresh(store.delete_record_ids(selected["record_id"].tolist()))

    columns = st.columns(2)
    experiment_id = columns[0].selectbox("Delete all rows for Experiment ID", [""] + sorted(all_records["experiment_id"].unique().tolist()))
    confirm_experiment = columns[0].checkbox("Confirm Experiment ID deletion")
    if columns[0].button("Delete Experiment ID rows", disabled=not experiment_id or not confirm_experiment):
        _delete_and_refresh(store.delete_where("experiment_id", experiment_id))

    image = columns[1].selectbox("Delete all rows for image filename", [""] + sorted(all_records["image_filename"].unique().tolist()))
    confirm_image = columns[1].checkbox("Confirm image deletion")
    if columns[1].button("Delete image rows", disabled=not image or not confirm_image):
        _delete_and_refresh(store.delete_where("image_filename", image))

    columns = st.columns(3)
    confirm_development = columns[0].checkbox("Confirm deletion of Development/Test records")
    if columns[0].button("Delete all Development/Test records", disabled=not confirm_development):
        _delete_and_refresh(store.delete_where("experiment_status", "Development / Test"))
    confirm_clear = columns[1].checkbox("Confirm clearing all evaluation records")
    if columns[1].button("Clear all evaluation records", disabled=not confirm_clear):
        _delete_and_refresh(store.clear())
    confirm_reset = columns[2].checkbox("Confirm evaluation-store reset")
    if columns[2].button("Reset evaluation store", disabled=not confirm_reset):
        _delete_and_refresh(store.reset(confirmed=True))


def _delete_and_refresh(result) -> None:
    count, experiment_ids = result
    affected = ", ".join(experiment_ids) if experiment_ids else "none"
    st.session_state.research_management_message = f"Deleted {count} rows. Affected Experiment IDs: {affected}. Summaries and charts were recomputed."
    st.rerun()


def _render_charts(summary: pd.DataFrame) -> None:
    st.markdown("#### Top-K Proposal Recall by Method")
    st.caption("Denominator: eligible anomaly-present images with verified or reviewer-estimated ground truth and a reviewer-confirmed useful rank.")
    if not has_valid_metric(summary,list(RECALL_CHART_COLUMNS.values())):
        st.info("No eligible anomaly-present experiments with verified or reviewer-estimated ground truth are available.")
    else:
        _grouped_chart(summary, RECALL_CHART_COLUMNS, "Top-K Proposal Recall by Method", "Proposal recall", percent=True)

    st.markdown("#### Proposal Review Outcomes by Method")
    st.caption("Means are shown by explicit state; not_reviewed is not treated as rejected.")
    _grouped_chart(summary, OUTCOME_CHART_COLUMNS, "Proposal Review Outcomes by Method", "Mean proposals per image")

    st.markdown("#### Annotation Efficiency by Method")
    st.caption("Acceptance uses accepted / (accepted + rejected). Uncertain and not_reviewed are excluded.")
    for title, column in EFFICIENCY_CHART_COLUMNS.items():
        percent = column == "annotation_acceptance_rate"
        _single_chart(summary, column, title, title, percent=percent)

    st.markdown("#### False Proposals per Reviewed Image")
    st.caption("False proposals are explicitly rejected proposals from manually reviewed methods only.")
    _single_chart(summary, FALSE_PROPOSAL_COLUMN, "False Proposals per Reviewed Image", "Mean rejected proposals", require_positive=True)


def _grouped_chart(summary, mapping, title, ylabel, percent=False):
    values = summary[["method"] + list(mapping.values())].copy()
    if not has_valid_metric(values,list(mapping.values())):
        st.info(f"{title}: N/A for the current records.")
        return
    figure, axis = plt.subplots(figsize=(9, chart_height(len(values))))
    x = np.arange(len(values)); width = .8 / len(mapping)
    for index, (label, column) in enumerate(mapping.items()):
        bars=axis.bar(x + (index - (len(mapping)-1)/2)*width, values[column], width, label=label)
        _label_bars(axis,bars,percent)
    axis.set_xticks(x, [METHOD_DISPLAY_NAMES.get(value,value) for value in values["method"]], rotation=14, ha="right")
    axis.set_xlabel("Proposal method"); axis.set_ylabel(ylabel); axis.set_title(title); axis.legend(title="Metric")
    if percent:
        axis.set_ylim(0, 1); axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    figure.tight_layout(); st.pyplot(figure); plt.close(figure)


def _single_chart(summary, column, title, ylabel, percent=False, require_positive=False):
    values = summary[["method", column]].dropna()
    if not has_valid_metric(values,[column],require_positive):
        st.info(EMPTY_METRIC_MESSAGES.get(column,f"{title}: no valid data are available."))
        return
    figure, axis = plt.subplots(figsize=(8, chart_height(len(values))))
    labels=[METHOD_DISPLAY_NAMES.get(value,value) for value in values["method"]]
    bars=axis.bar(labels, values[column])
    axis.set_xlabel("Proposal method"); axis.set_ylabel(ylabel); axis.set_title(title)
    axis.tick_params(axis="x", rotation=18)
    if percent:
        axis.set_ylim(0, 1); axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    _label_bars(axis,bars,percent)
    figure.tight_layout(); st.pyplot(figure); plt.close(figure)


def chart_height(method_count:int)->float:
    if method_count<=1:return 2.6
    if method_count<=4:return 3.8
    return 4.8


def has_valid_metric(frame:pd.DataFrame,columns:list[str],require_positive=False)->bool:
    if frame.empty or not all(column in frame for column in columns):return False
    values=frame[columns].apply(pd.to_numeric,errors="coerce")
    if values.dropna(how="all").empty:return False
    return bool((values>0).any().any()) if require_positive else True


def _label_bars(axis,bars,percent=False):
    for bar in bars:
        value=bar.get_height()
        if np.isnan(value):continue
        label=f"{value:.0%}" if percent else f"{value:.2f}"
        axis.text(bar.get_x()+bar.get_width()/2,value,label,ha="center",va="bottom",fontsize=8)


def _display_na(frame: pd.DataFrame):
    return frame.style.format(na_rep="N/A", precision=3)


def _render_legacy_migration(store: ExperimentStore, legacy_path: Path) -> None:
    st.markdown("### Legacy Record Migration")
    rows = load_legacy_records(legacy_path)
    legacy_indices = legacy_record_indices(rows)
    if not legacy_indices:
        st.info("No legacy records missing record_id, review_status, not_reviewed, or experiment_status were detected.")
        return
    legacy = pd.DataFrame([rows[index] for index in legacy_indices])
    st.warning(f"Detected {len(legacy)} legacy rows. Historical files are not changed automatically.")
    selection = st.dataframe(legacy, width="stretch", on_select="rerun", selection_mode="multi-row", key="legacy_record_selection")
    selected_positions = list(selection.selection.rows)
    selected_indices = [legacy_indices[position] for position in selected_positions]
    st.button("Keep legacy records unchanged", help="No migration or deletion is performed.")
    selected_baselines = [index for index in selected_indices if str(rows[index].get("method", "")) != "refined contextual method"]
    if st.button("Migrate selected unreviewed baseline rows to not_reviewed", disabled=not selected_baselines):
        count = migrate_legacy_records(rows, selected_baselines, store)
        st.success(f"Migrated {count} legacy rows into SQLite; the legacy JSON remains unchanged.")
    confirm = st.checkbox("Confirm deletion of selected legacy records")
    if st.button("Delete selected legacy records", disabled=not selected_indices or not confirm):
        count = delete_legacy_records(legacy_path, rows, selected_indices)
        st.success(f"Deleted {count} selected legacy rows from the legacy JSON file.")

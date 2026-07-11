"""Streamlit renderer for persistent Research Evaluation workflows."""

from __future__ import annotations

from datetime import date, datetime
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


def render_research_evaluation(
    output_dir: Path,
    image_name: str | None,
    annotations,
    proposal_result,
    feature_maps,
    review_start_time: str | None,
    review_completion_time: str | None,
) -> None:
    st.subheader("Research Evaluation")
    database_path = Path(os.environ.get("STRUCTVISION_RESEARCH_DB", output_dir / "research_evaluation.sqlite3"))
    legacy_json_path = Path(os.environ.get("STRUCTVISION_LEGACY_EXPERIMENT_JSON", output_dir / "research_experiment_results.json"))
    store = ExperimentStore(database_path)
    _render_recording(
        store, image_name, annotations, proposal_result, feature_maps,
        review_start_time, review_completion_time,
    )
    st.divider()
    _render_dashboard_and_management(store)
    st.divider()
    _render_legacy_migration(store, legacy_json_path)


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
        st.dataframe(preview[IMAGE_TABLE_COLUMNS], use_container_width=True)
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


def _render_dashboard_and_management(store: ExperimentStore) -> None:
    st.markdown("### Manage Experiments")
    all_records = store.dataframe()
    if all_records.empty:
        st.info("No SQLite experiment records are stored yet.")
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
            image_table[display_columns], use_container_width=True, hide_index=True,
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
        st.dataframe(_display_na(summary), use_container_width=True, hide_index=True)
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
    if summary.empty or summary[list(RECALL_CHART_COLUMNS.values())].dropna(how="all").empty:
        st.info("No eligible anomaly-present experiments are available for Top-K recall.")
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
    _single_chart(summary, FALSE_PROPOSAL_COLUMN, "False Proposals per Reviewed Image", "Mean rejected proposals")


def _grouped_chart(summary, mapping, title, ylabel, percent=False):
    values = summary[["method"] + list(mapping.values())].copy()
    if values[list(mapping.values())].dropna(how="all").empty:
        st.info(f"{title}: N/A for the current records.")
        return
    figure, axis = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(values)); width = .8 / len(mapping)
    for index, (label, column) in enumerate(mapping.items()):
        axis.bar(x + (index - (len(mapping)-1)/2)*width, values[column], width, label=label)
    axis.set_xticks(x, values["method"], rotation=18, ha="right")
    axis.set_xlabel("Proposal method"); axis.set_ylabel(ylabel); axis.set_title(title); axis.legend(title="Metric")
    if percent:
        axis.set_ylim(0, 1); axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    figure.tight_layout(); st.pyplot(figure); plt.close(figure)


def _single_chart(summary, column, title, ylabel, percent=False):
    values = summary[["method", column]].dropna()
    if values.empty:
        st.info(f"{title}: N/A for the current records.")
        return
    figure, axis = plt.subplots(figsize=(9, 3.8))
    axis.bar(values["method"], values[column], label=title)
    axis.set_xlabel("Proposal method"); axis.set_ylabel(ylabel); axis.set_title(title); axis.legend()
    axis.tick_params(axis="x", rotation=18)
    if percent:
        axis.set_ylim(0, 1); axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    figure.tight_layout(); st.pyplot(figure); plt.close(figure)


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
    selection = st.dataframe(legacy, use_container_width=True, on_select="rerun", selection_mode="multi-row", key="legacy_record_selection")
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

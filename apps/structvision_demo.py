"""Thin Streamlit client for public StructVision demonstration interfaces."""

from __future__ import annotations

import streamlit as st

from structvision import (
    CLASSICAL_METHOD,
    DEFAULT_METHOD,
    EVIDENCE_ROWS,
    FIXTURE_LABELS,
    HYBRID_METHOD,
    METHOD_STATUSES,
    PATCHCORE_METHOD,
    OperationalStorageContext,
    StorageConfigurationError,
    LearnedEnvironmentUnavailableError,
    LearnedRuntimePaths,
    analyse_demonstration_image,
    analysis_json_bytes,
    annotated_png_bytes,
    binary_mask_png_bytes,
    candidate_crop,
    candidate_mask,
    candidate_rows,
    decode_image_bytes,
    demonstration_fixture,
    export_payload,
    method_availability,
    pipeline_stages,
    proposal_csv_bytes,
    render_anomaly_overlay,
    render_overlay,
    technical_summary_bytes,
)


st.set_page_config(
    page_title="StructVision-AI — Algorithm Demonstration",
    page_icon="🔬",
    layout="wide",
)

METHOD_LABELS = {
    CLASSICAL_METHOD: "Frozen classical baseline — stable default",
    PATCHCORE_METHOD: "PatchCore — protected development baseline",
    HYBRID_METHOD: "Proposal-guided hybrid — rejected development candidate",
}


def _rgb(image_bgr):
    return image_bgr[..., ::-1]


def _display_statuses(runtime: LearnedRuntimePaths) -> None:
    for status in METHOD_STATUSES:
        availability = method_availability(status.method_id, runtime)
        with st.container(border=True):
            left, right = st.columns([3, 2])
            left.markdown(f"**{status.method_id}**")
            left.caption(f"Status: {status.status}")
            left.write(status.operational_role)
            right.write("Available" if availability.available else "Execution disabled")
            if not availability.available:
                right.caption(availability.message)
            right.caption(status.evidence_limit)


def _render_candidate_table(analysis) -> tuple[dict[str, object], ...]:
    rows = candidate_rows(analysis)
    display = []
    for row in rows:
        item = dict(row)
        item["rank"] = "N/A" if item["rank"] is None else item["rank"]
        for key, value in tuple(item.items()):
            if value is None:
                item[key] = "N/A"
        display.append(item)
    st.dataframe(display, hide_index=True, width="stretch")
    return rows


def _analysis_workspace(analysis) -> None:
    st.subheader("Analysis output")
    status_cols = st.columns(4)
    status_cols[0].metric("Processing status", "Completed")
    status_cols[1].metric("Analysed width", analysis.image_shape[1])
    status_cols[2].metric("Analysed height", analysis.image_shape[0])
    status_cols[3].metric(
        "Selected proposals",
        sum(bool(row["selected"]) for row in candidate_rows(analysis)),
    )
    st.caption(
        f"Method: {analysis.method_id} · Status: {analysis.method.status} · "
        f"Configuration: {analysis.configuration_hash}"
    )
    st.warning(analysis.score_semantics)

    input_tab, overlay_tab, evidence_tab, candidates_tab, timing_tab, export_tab = st.tabs(
        (
            "Input and overlay",
            "Anomaly evidence",
            "Proposal evidence",
            "Candidate detail",
            "Timing and identity",
            "Explicit exports",
        )
    )
    with input_tab:
        left, right = st.columns(2)
        left.image(
            _rgb(analysis.input_image.image_bgr),
            caption=(
                f"Input held in memory · {analysis.input_image.width}×"
                f"{analysis.input_image.height} · {analysis.input_image.colour_handling}"
            ),
            width="stretch",
        )
        right.image(
            _rgb(render_overlay(analysis)),
            caption="Direct returned masks and half-open mask-derived boxes",
            width="stretch",
        )
        st.json(dict(analysis.coordinate_mapping))
    with overlay_tab:
        anomaly = render_anomaly_overlay(analysis)
        if anomaly is None:
            st.info("Not exposed by the current frozen API.")
        else:
            st.image(
                _rgb(anomaly),
                caption="Display-normalised anomaly evidence; no proposal boundary is altered",
                width="stretch",
            )
    with evidence_tab:
        rows = _render_candidate_table(analysis)
        if analysis.method_id == HYBRID_METHOD:
            st.caption(
                "The table retains selected and rejected classical candidates. "
                "Rejected rows have no final rank."
            )
        if not rows:
            st.info("The method returned no proposal at its frozen operating point.")
    with candidates_tab:
        rows = candidate_rows(analysis)
        if not rows:
            st.info("No candidate is available for detailed inspection.")
        else:
            candidate_id = st.selectbox(
                "Candidate",
                [str(row["proposal_id"]) for row in rows],
                format_func=lambda value: next(
                    (
                        f"{'rank ' + str(row['rank']) if row['rank'] is not None else 'rejected'}"
                        f" · {value}"
                    )
                    for row in rows
                    if row["proposal_id"] == value
                ),
            )
            selected = next(row for row in rows if row["proposal_id"] == candidate_id)
            col1, col2, col3 = st.columns(3)
            col1.image(
                _rgb(render_overlay(analysis, candidate_id=candidate_id)),
                caption="Single-candidate full image",
                width="stretch",
            )
            col2.image(
                _rgb(candidate_crop(analysis, candidate_id)),
                caption="Half-open box crop",
                width="stretch",
            )
            col3.image(
                candidate_mask(analysis, candidate_id),
                caption="Returned binary mask",
                clamp=True,
                width="stretch",
            )
            st.json({
                key: ("N/A" if value is None else value)
                for key, value in selected.items()
            })
    with timing_tab:
        timing = getattr(analysis.result, "timing_breakdown_seconds", None)
        if timing is None:
            inference = getattr(analysis.result, "inference_seconds", None)
            timing = () if inference is None else (("inference", inference),)
        st.dataframe(
            [{"stage": name, "seconds": value} for name, value in timing],
            hide_index=True,
            width="stretch",
        )
        st.json({
            "implementation_identity": analysis.method_id,
            "implementation_version": analysis.method.version,
            "configuration_hash": analysis.configuration_hash,
            "artifact_identities": export_payload(analysis)["analysis"]["artifact_identities"],
            "image_anomaly_score": export_payload(analysis)["analysis"]["image_anomaly_score"],
            "input_image_hash": analysis.input_image.encoded_sha256,
            "normalised_input_hash": analysis.input_hash,
            "development_status": analysis.method.status,
            "warnings": list(analysis.warnings),
        })
    with export_tab:
        st.caption(
            "Nothing is persisted by analysis. A file is transferred only after an explicit download click."
        )
        stem = f"structvision-{analysis.input_image.encoded_sha256[:12]}"
        st.download_button(
            "Download analysis JSON",
            analysis_json_bytes(analysis),
            f"{stem}.json",
            "application/json",
        )
        st.download_button(
            "Download proposal table CSV",
            proposal_csv_bytes(analysis),
            f"{stem}-proposals.csv",
            "text/csv",
        )
        st.download_button(
            "Download annotated image PNG",
            annotated_png_bytes(analysis),
            f"{stem}-annotated.png",
            "image/png",
        )
        st.download_button(
            "Download concise technical summary",
            technical_summary_bytes(analysis),
            f"{stem}-summary.txt",
            "text/plain",
        )
        rows = candidate_rows(analysis)
        if rows:
            mask_id = st.selectbox(
                "Binary mask to export",
                [str(row["proposal_id"]) for row in rows],
                key="mask_export_id",
            )
            st.download_button(
                "Download selected binary mask PNG",
                binary_mask_png_bytes(analysis, mask_id),
                f"{stem}-{mask_id}-mask.png",
                "image/png",
            )


def main() -> None:
    try:
        storage_context = OperationalStorageContext.discover()
        runtime = LearnedRuntimePaths.from_environment(storage_context)
    except StorageConfigurationError:
        st.title("StructVision-AI")
        st.error(
            "External storage configuration is missing, malformed, or unsafe. "
            "No analysis or protected-resource access was attempted."
        )
        return
    st.title("StructVision-AI")
    st.caption("Live inspection console and technical-validation interface")
    st.info(
        "This client is an algorithmic inspection interface. It proposes regions for "
        "technical review; it is not a defect classifier, engineering diagnosis, or "
        "image-generation product."
    )

    (
        overview,
        analyse,
        pipeline,
        proposal_evidence,
        comparison,
        architecture,
        research,
        data_contract,
        reproducibility,
    ) = st.tabs(
        (
            "System Overview",
            "Analyse an Image",
            "Algorithm Pipeline",
            "Proposal Evidence",
            "Method Comparison",
            "Source-Code Architecture",
            "Research Evidence",
            "Data Integration Contract",
            "Reproducibility and Limitations",
        )
    )

    with overview:
        st.subheader("Operational policy")
        st.write(
            "The stable frozen classical baseline is the default because it is reusable "
            "in the base environment and retains the strongest current sensitivity evidence."
        )
        _display_statuses(runtime)
    with analyse:
        source = st.radio(
            "Image source",
            ("Upload one image", "Select deterministic demonstration fixture"),
            horizontal=True,
        )
        decoded = None
        if source == "Upload one image":
            uploaded = st.file_uploader(
                "PNG, JPEG, or TIFF",
                type=("png", "jpg", "jpeg", "tif", "tiff"),
                accept_multiple_files=False,
            )
            alpha = st.selectbox(
                "Alpha handling (applied only when an alpha channel is present)",
                ("composite_white", "composite_black", "drop"),
            )
            if uploaded is not None:
                try:
                    decoded = decode_image_bytes(
                        uploaded.getvalue(),
                        filename=uploaded.name,
                        alpha_handling=alpha,
                    )
                except Exception as error:
                    st.error(str(error))
        else:
            fixture = st.selectbox("Fixture", FIXTURE_LABELS)
            decoded = demonstration_fixture(fixture)
            st.warning(
                "Synthetic demonstration fixture only; excluded from research evaluation "
                "and not presented as real inspection evidence."
            )

        available_methods = [
            status.method_id
            for status in METHOD_STATUSES
            if method_availability(status.method_id, runtime).available
        ]
        method_id = st.selectbox(
            "Method",
            available_methods,
            index=available_methods.index(DEFAULT_METHOD),
            format_func=lambda value: METHOD_LABELS[value],
        )
        for status in METHOD_STATUSES:
            if status.method_id not in available_methods:
                st.caption(
                    f"{METHOD_LABELS[status.method_id]} — "
                    f"{method_availability(status.method_id, runtime).message}"
                )
        if st.button("Run local analysis", type="primary", disabled=decoded is None):
            try:
                with st.spinner("Running the selected immutable method locally…"):
                    st.session_state["live_analysis"] = analyse_demonstration_image(
                        decoded,
                        method_id=method_id,
                        runtime=runtime,
                    )
            except LearnedEnvironmentUnavailableError as error:
                st.error(str(error))
            except Exception as error:
                st.error(f"{type(error).__name__}: {error}")
        current = st.session_state.get("live_analysis")
        if current is not None:
            _analysis_workspace(current)
    with pipeline:
        selected_pipeline = st.selectbox(
            "Method pipeline",
            [status.method_id for status in METHOD_STATUSES],
            format_func=lambda value: METHOD_LABELS[value],
        )
        st.dataframe(
            pipeline_stages(selected_pipeline),
            hide_index=True,
            width="stretch",
        )
    with proposal_evidence:
        st.write(
            "Returned masks define the scientific region boundary. Bounding boxes are "
            "recomputed from those masks using half-open coordinates "
            "`(x_min, y_min, x_max, y_max)`."
        )
        st.write(
            "Classical scores are review heuristics. PatchCore values are distances. "
            "Hybrid values are fixed linear rank scores. None is a probability."
        )
        current = st.session_state.get("live_analysis")
        if current is None:
            st.info("Run an image to inspect concrete candidate evidence.")
        else:
            _render_candidate_table(current)
    with comparison:
        st.subheader("Current synthetic development evidence")
        st.dataframe(EVIDENCE_ROWS, hide_index=True, width="stretch")
        st.error(
            "The hybrid is a rejected development candidate. Its burden and precision "
            "improvements do not override the failed sensitivity-preservation gate."
        )
    with architecture:
        st.code(
            "Technical demonstration client\n"
            "  ↓ public structvision demonstration facade\n"
            "  ↓ public typed detector APIs\n"
            "  ↓ protected immutable implementations\n"
            "  ↓ typed in-memory results\n",
            language="text",
        )
        st.write(
            "Review: `src/structvision/api.py`, `src/structvision/demonstration.py`, "
            "`src/structvision/classical.py`, `src/structvision/normal_feature/`, "
            "`src/structvision/hybrid/`, and `scientific_contract/`."
        )
    with research:
        st.write(
            "The current evidence is synthetic and development-only. Its strength is the "
            "transparent complementary-method trade-off, disciplined rejection of the "
            "hybrid, reusable local APIs, and protected path to future data."
        )
        st.write(
            "Hybrid primary micro sensitivity fell from 0.770833 to 0.750000. "
            "The 0.020833 loss exceeded the fixed 0.02 margin by approximately 0.000833; "
            "image-level sensitivity also decreased."
        )
    with data_contract:
        st.write(
            "Future private data connects through a dataset adapter outside detector logic. "
            "The adapter supplies immutable IDs, content hashes, explicit colour and "
            "annotation semantics, acquisition groups, licensing/confidentiality metadata, "
            "and split-lock identity."
        )
        st.warning(
            "This demonstration does not read private collaborator data, modify a registry, upload "
            "images, or create evaluation rows."
        )
    with reproducibility:
        st.write(
            "Base operation is offline after installation, uses no API key, and performs no "
            "implicit repository write. Learned methods require exact local dependencies, "
            "verified immutable artifacts, and an already cached official weight."
        )
        st.write(
            "Known limitations include synthetic-only evidence, no real-world transfer "
            "validation, uncalibrated score semantics, memory pressure on large images, "
            "no physical scale, and no engineering diagnosis."
        )
        st.code(
            "structvision-analyse --input inspection.png --method classical\n"
            "python -m streamlit run apps/structvision_demo.py",
            language="bash",
        )


if __name__ == "__main__":
    main()

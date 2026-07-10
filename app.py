"""Streamlit application for marine structural defect inspection."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2
import pandas as pd
import streamlit as st
from PIL import Image

from config import OUTPUT_DIR, REPORT_DIR, UPLOAD_DIR
from detect import run_detection
from explain import build_engineering_summary, recommend_actions
from preprocess import apply_preprocessing, build_preview_grid
from report import generate_pdf_report
from severity import estimate_overall_severity


st.set_page_config(
    page_title="Marine Structural Defect Inspection",
    page_icon="⚓",
    layout="wide",
)


def _save_upload(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix.lower() or ".png"
    safe_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}{suffix}"
    upload_path = UPLOAD_DIR / safe_name
    upload_path.write_bytes(uploaded_file.getbuffer())
    return upload_path


def _read_image(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError("Uploaded file could not be read as an image.")
    return image


def main() -> None:
    st.title("AI-Based Visual Inspection and Defect Severity Analysis for Marine Structural Components")
    st.caption(
        "Computer vision workflow for hull plates, offshore members, pipelines, welded joints, "
        "coated surfaces, and metallic panels."
    )

    with st.sidebar:
        st.header("Inspection Controls")
        uploaded_file = st.file_uploader("Upload inspection image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
        st.subheader("Preprocessing")
        resize_width = st.slider("Resize width", min_value=640, max_value=1600, value=1024, step=64)
        use_denoise = st.checkbox("Denoise", value=True)
        use_clahe = st.checkbox("CLAHE contrast enhancement", value=True)
        use_sharpen = st.checkbox("Sharpen", value=False)
        show_previews = st.checkbox("Show preprocessing previews", value=True)
        st.subheader("Detection")
        confidence = st.slider("YOLO confidence threshold", min_value=0.10, max_value=0.90, value=0.35, step=0.05)
        force_classical = st.checkbox("Force classical CV demo mode", value=False)

    st.markdown(
        "Manual inspection in marine environments is time-consuming and subjective. This prototype "
        "combines visual defect localization, severity scoring, and engineering-style reporting to "
        "support early inspection triage."
    )

    if uploaded_file is None:
        st.info("Upload an inspection image to begin analysis.")
        st.markdown(
            "Relevant targets include corroded hull plates, coating breakdown, crack-like discontinuities, "
            "weld defects, dents, deformation, pitting, scratches, and general surface anomalies."
        )
        return

    try:
        upload_path = _save_upload(uploaded_file)
        raw_image = _read_image(upload_path)
    except Exception as exc:
        st.error(f"Could not process upload: {exc}")
        return

    processed = apply_preprocessing(
        raw_image,
        resize_width=resize_width,
        denoise=use_denoise,
        clahe=use_clahe,
        sharpen=use_sharpen,
    )

    detection = run_detection(
        image=processed,
        original_name=uploaded_file.name,
        output_dir=OUTPUT_DIR,
        confidence_threshold=confidence,
        force_classical=force_classical,
    )

    severity = estimate_overall_severity(detection.defects, processed.shape[:2])
    interpretation = build_engineering_summary(detection.defects, severity)
    actions = recommend_actions(detection.defects, severity)

    top_cols = st.columns([1.05, 1])
    with top_cols[0]:
        st.subheader("Uploaded Image")
        st.image(Image.open(upload_path), use_container_width=True)
    with top_cols[1]:
        st.subheader("Annotated Output")
        st.image(detection.annotated_image_path.as_posix(), use_container_width=True)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Detection Mode", detection.mode)
    metric_cols[1].metric("Defect Count", len(detection.defects))
    metric_cols[2].metric("Severity Score", f"{severity.score:.1f}/100")
    metric_cols[3].metric("Severity Level", severity.label)

    if detection.message:
        st.warning(detection.message)

    st.subheader("Defect Summary")
    if detection.defects:
        table = pd.DataFrame([defect.to_table_row() for defect in detection.defects])
        st.dataframe(table, use_container_width=True, hide_index=True)
    else:
        st.success("No visible defect region was identified by the selected detection mode.")

    st.subheader("Engineering Interpretation")
    st.write(interpretation)

    st.subheader("Recommended Inspection Action")
    for action in actions:
        st.write(f"- {action}")

    if show_previews:
        st.subheader("Preprocessing Preview")
        previews = build_preview_grid(processed)
        preview_cols = st.columns(len(previews))
        for col, (title, preview) in zip(preview_cols, previews.items()):
            with col:
                st.caption(title)
                st.image(preview, channels="BGR" if preview.ndim == 3 else "GRAY", use_container_width=True)

    report_path = generate_pdf_report(
        report_dir=REPORT_DIR,
        image_name=uploaded_file.name,
        detection=detection,
        severity=severity,
        interpretation=interpretation,
        actions=actions,
    )

    download_cols = st.columns(2)
    with download_cols[0]:
        st.download_button(
            "Download Annotated Image",
            data=detection.annotated_image_path.read_bytes(),
            file_name=detection.annotated_image_path.name,
            mime="image/png",
        )
    with download_cols[1]:
        st.download_button(
            "Download PDF Inspection Report",
            data=report_path.read_bytes(),
            file_name=report_path.name,
            mime="application/pdf",
        )

    st.caption(
        "Disclaimer: This is an AI-assisted visual inspection prototype and should not replace "
        "certified marine or structural inspection."
    )


if __name__ == "__main__":
    main()

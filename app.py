"""Streamlit interface for StructVision-AI."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2
import pandas as pd
import streamlit as st
from PIL import Image

from config import DEFAULT_LABEL_CLASSES, OUTPUT_DIR, PROJECT_SUBTITLE, PROJECT_TITLE, REPORT_DIR, UPLOAD_DIR
from dataset_export import export_dataset
from feature_extraction import extract_feature_maps, save_feature_maps
from labeling import build_annotation
from preprocess import apply_preprocessing
from region_proposal import create_region_crops, propose_regions
from report import generate_pdf_report
from yolo_inference import run_yolo_inference


st.set_page_config(page_title="StructVision-AI", layout="wide")


def save_upload(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix.lower() or ".png"
    safe_stem = Path(uploaded_file.name).stem.replace(" ", "_")[:48] or "image"
    path = UPLOAD_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_stem}_{uuid4().hex[:6]}{suffix}"
    path.write_bytes(uploaded_file.getbuffer())
    return path


def load_cv_image(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path.name}")
    return image


def init_state() -> None:
    defaults = {
        "image_path": None,
        "image_name": None,
        "processed": None,
        "feature_maps": None,
        "feature_paths": {},
        "proposal_result": None,
        "annotations": [],
        "yolo_result": None,
        "preprocess_settings": {},
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def run_analysis(uploaded_file, settings: dict[str, int | bool]) -> None:
    image_path = save_upload(uploaded_file)
    raw = load_cv_image(image_path)
    processed = apply_preprocessing(
        raw,
        resize_width=int(settings["resize_width"]),
        denoise=bool(settings["denoise"]),
        clahe=bool(settings["clahe"]),
        sharpen=bool(settings["sharpen"]),
        brightness=int(settings["brightness"]),
        contrast=int(settings["contrast"]),
    )
    image_stem = image_path.stem
    feature_maps = extract_feature_maps(
        processed,
        edge_sensitivity=int(settings["edge_sensitivity"]),
        texture_sensitivity=int(settings["texture_sensitivity"]),
        color_sensitivity=int(settings["color_sensitivity"]),
        threshold_level=int(settings["threshold_level"]),
    )
    feature_paths = save_feature_maps(feature_maps, image_stem)
    proposal_result = propose_regions(
        processed,
        feature_maps,
        image_stem=image_stem,
        min_area=int(settings["min_area"]),
        max_regions=int(settings["max_regions"]),
        min_relative_area=float(settings["min_relative_area"]),
        max_relative_area=float(settings["max_relative_area"]),
    )
    yolo_result = run_yolo_inference(processed, image_stem, confidence_threshold=float(settings["yolo_confidence"]))

    st.session_state.image_path = image_path
    st.session_state.image_name = uploaded_file.name
    st.session_state.processed = processed
    st.session_state.feature_maps = feature_maps
    st.session_state.feature_paths = feature_paths
    st.session_state.proposal_result = proposal_result
    st.session_state.yolo_result = yolo_result
    st.session_state.annotations = []
    st.session_state.preprocess_settings = settings


def main() -> None:
    init_state()
    st.title(PROJECT_TITLE)
    st.caption(PROJECT_SUBTITLE)

    with st.sidebar:
        st.header("Input")
        uploaded_files = st.file_uploader(
            "Upload image(s)",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
            accept_multiple_files=True,
        )
        video_file = st.file_uploader("Optional video upload", type=["mp4", "mov", "avi"], accept_multiple_files=False)
        if video_file is not None:
            st.info("Video upload is recognized. Frame extraction is future-ready and not executed automatically in this prototype.")

        st.header("Preprocessing")
        settings = {
            "resize_width": st.slider("Resize width", 512, 1800, 1024, 64),
            "brightness": st.slider("Brightness", -80, 80, 0, 5),
            "contrast": st.slider("Contrast", -80, 120, 0, 5),
            "denoise": st.checkbox("Denoise", value=True),
            "clahe": st.checkbox("CLAHE enhancement", value=True),
            "sharpen": st.checkbox("Sharpen", value=False),
            "edge_sensitivity": st.slider("Edge sensitivity", 30, 220, 100, 5),
            "texture_sensitivity": st.slider("Texture sensitivity", 5, 120, 35, 5),
            "color_sensitivity": st.slider("Color sensitivity", 5, 120, 35, 5),
            "threshold_level": st.slider("Threshold level", 20, 235, 128, 5),
            "min_area": st.slider("Minimum region area", 50, 5000, 250, 50),
            "min_relative_area": st.slider("Minimum relative area", 0.0001, 0.0200, 0.0002, 0.0001, format="%.4f"),
            "max_relative_area": st.slider("Maximum relative area", 0.10, 0.95, 0.85, 0.05),
            "max_regions": st.slider("Maximum regions", 3, 60, 20, 1),
            "yolo_confidence": st.slider("YOLO confidence", 0.10, 0.90, 0.35, 0.05),
        }

        analyze = st.button("Analyze Selected Image", type="primary", use_container_width=True)

    if analyze:
        if not uploaded_files:
            st.error("Upload at least one image before analysis.")
        else:
            with st.spinner("Extracting features and proposing candidate regions..."):
                run_analysis(uploaded_files[0], settings)

    tabs = st.tabs(
        [
            "Overview",
            "Image Analysis",
            "Feature Maps",
            "Region Proposals",
            "Human Review / Labeling",
            "Dataset Export",
            "Report Generation",
            "Future Model Training",
        ]
    )

    with tabs[0]:
        st.subheader("Purpose")
        st.write(
            "StructVision-AI analyzes structural, product, component, or surface images before labeled data exists. "
            "It proposes visually significant anomaly candidates, creates segmentation-ready masks, supports human review, "
            "and exports annotations for future YOLO training."
        )
        st.markdown(
            "Raw image → preprocessing → feature extraction → anomaly region proposal → mask output → visual priority scoring "
            "→ human review → dataset export → future YOLO/SAM training → inspection report."
        )
        st.info("Before a trained model is present, all regions are candidate proposals, not certified defect predictions.")

    with tabs[1]:
        st.subheader("Image Analysis")
        if st.session_state.image_path is None:
            st.info("Upload and analyze an image from the sidebar.")
        else:
            cols = st.columns(2)
            cols[0].caption("Uploaded image")
            cols[0].image(Image.open(st.session_state.image_path), use_container_width=True)
            cols[1].caption("Preprocessed analysis image")
            cols[1].image(st.session_state.processed, channels="BGR", use_container_width=True)
            yolo = st.session_state.yolo_result
            st.write(yolo.message)
            if yolo.available and yolo.annotated_path:
                st.image(yolo.annotated_path.as_posix(), caption="Trained YOLO predictions", use_container_width=True)
                st.dataframe(pd.DataFrame([pred.to_row() for pred in yolo.predictions]), use_container_width=True)

    with tabs[2]:
        st.subheader("Feature Maps")
        if st.session_state.feature_maps is None:
            st.info("Run analysis to generate feature maps.")
        else:
            fmap_items = list(st.session_state.feature_maps.as_dict().items())
            for row_start in range(0, len(fmap_items), 3):
                cols = st.columns(3)
                for col, (name, fmap) in zip(cols, fmap_items[row_start : row_start + 3]):
                    col.caption(name)
                    if fmap.ndim == 2:
                        col.image(fmap, clamp=True, use_container_width=True)
                    else:
                        col.image(fmap, channels="BGR", use_container_width=True)

    with tabs[3]:
        st.subheader("Region Proposals")
        proposal_result = st.session_state.proposal_result
        if proposal_result is None:
            st.info("Run analysis to create region proposals.")
        else:
            st.image(proposal_result.overlay_path.as_posix(), caption="Highlighted visual anomaly candidates", use_container_width=True)
            diagnostic_cols = st.columns(5)
            diagnostics = proposal_result.diagnostics
            diagnostic_cols[0].metric("Raw components", diagnostics.raw_components)
            diagnostic_cols[1].metric("After filtering", diagnostics.after_filtering)
            diagnostic_cols[2].metric("After merging", diagnostics.after_merging)
            diagnostic_cols[3].metric("Heat threshold", f"{diagnostics.heatmap_threshold:.1f}")
            median_score = float(pd.Series(diagnostics.score_distribution).median()) if diagnostics.score_distribution else 0.0
            diagnostic_cols[4].metric("Median score", f"{median_score:.1f}")
            st.dataframe(pd.DataFrame([proposal.to_row() for proposal in proposal_result.proposals]), use_container_width=True)
            st.caption("Algorithm comparison")
            comparison_cols = st.columns(3)
            for col, (name, path) in zip(comparison_cols, proposal_result.comparison_paths.items()):
                col.image(path.as_posix(), caption=f"{name} | {proposal_result.comparison_counts[name]} regions", use_container_width=True)
            st.download_button(
                "Download Combined Binary Mask",
                proposal_result.combined_mask_path.read_bytes(),
                file_name=proposal_result.combined_mask_path.name,
                mime="image/png",
            )

    with tabs[4]:
        st.subheader("Human Review / Labeling")
        proposal_result = st.session_state.proposal_result
        if proposal_result is None:
            st.info("Run analysis before reviewing candidate regions.")
        elif not proposal_result.proposals:
            st.warning("No candidate regions were proposed with the current filters.")
        else:
            st.write("Assign candidate labels for dataset creation. These labels are human review metadata, not model predictions.")
            annotations = []
            for proposal in proposal_result.proposals:
                with st.expander(f"{proposal.region_id} | {proposal.priority.label} | score {proposal.priority.score}", expanded=False):
                    crop_cols = st.columns(3)
                    crops = create_region_crops(st.session_state.processed, st.session_state.feature_maps, proposal)
                    for col, (name, crop) in zip(crop_cols, crops.items()):
                        col.image(crop, channels="BGR", caption=name, use_container_width=False)
                    st.caption(proposal.explanation)
                    st.json({
                        "edge_density": round(proposal.edge_density, 3), "texture_variation": round(proposal.texture_score, 3),
                        "colour_difference": round(proposal.color_variation_score, 3), "gradient_strength": round(proposal.gradient_strength, 3),
                        "entropy": round(proposal.entropy, 3), "mask_stability": round(proposal.mask_stability, 3),
                    })
                    accepted = st.checkbox("Accept region", value=proposal.priority.label in {"High", "Review Required"}, key=f"accept_{proposal.region_id}")
                    label = st.selectbox("Candidate label", DEFAULT_LABEL_CLASSES, key=f"label_{proposal.region_id}")
                    custom = st.text_input("Optional custom label", key=f"custom_{proposal.region_id}")
                    notes = st.text_area("Notes", key=f"notes_{proposal.region_id}", height=70)
                    final_label = custom.strip() if custom.strip() else label
                    annotations.append(build_annotation(st.session_state.image_name, proposal, accepted, final_label, notes))
            if st.button("Save Review Metadata", type="primary"):
                st.session_state.annotations = annotations
                st.success(f"Saved {len(annotations)} reviewed region records in session state.")

    with tabs[5]:
        st.subheader("Dataset Export")
        if st.session_state.image_path is None:
            st.info("Analyze and review an image before exporting.")
        elif not st.session_state.annotations:
            st.warning("Save review metadata first. Export uses accepted regions and their candidate labels.")
        else:
            accepted = [ann for ann in st.session_state.annotations if ann.accepted and ann.label != "ignore"]
            st.metric("Accepted candidate regions", len(accepted))
            if st.button("Export Reviewed Dataset Files", type="primary"):
                paths = export_dataset(
                    st.session_state.image_path,
                    st.session_state.processed.shape[:2],
                    st.session_state.annotations,
                )
                st.success("Dataset export completed.")
                st.json({name: path.as_posix() for name, path in paths.items()})

    with tabs[6]:
        st.subheader("Report Generation")
        if st.session_state.proposal_result is None:
            st.info("Run analysis before generating a report.")
        else:
            if st.button("Generate PDF Report", type="primary"):
                report_path = generate_pdf_report(
                    report_dir=REPORT_DIR,
                    image_name=st.session_state.image_name,
                    preprocessing_settings=st.session_state.preprocess_settings,
                    feature_paths=st.session_state.feature_paths,
                    proposal_result=st.session_state.proposal_result,
                    annotations=st.session_state.annotations,
                    yolo_result=st.session_state.yolo_result,
                )
                st.success(f"Report generated: {report_path.name}")
                st.download_button("Download Report", report_path.read_bytes(), file_name=report_path.name, mime="application/pdf")

    with tabs[7]:
        st.subheader("Future Model Training")
        st.write(
            "After reviewing and exporting enough labeled candidate regions, train YOLO detection or segmentation with Ultralytics. "
            "The app will automatically show trained inference separately when `models/best.pt` exists."
        )
        st.code("python train.py --data datasets/data.yaml --task detect --model yolo11n.pt --epochs 80 --imgsz 640", language="bash")
        st.code("python train.py --data datasets/data.yaml --task segment --model yolo11n-seg.pt --epochs 80 --imgsz 640", language="bash")
        st.info("SAM/SAM2 integration is future-ready: use proposed boxes as prompts, then replace rectangular masks with refined masks.")


if __name__ == "__main__":
    main()

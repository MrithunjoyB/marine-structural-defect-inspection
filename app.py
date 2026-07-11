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
from evaluation import evaluate_method, evaluation_tables
from feature_extraction import extract_feature_maps, save_feature_maps
from labeling import build_annotation
from preprocess import apply_preprocessing
from region_proposal import AblationConfig, _components, correct_region_mask, create_region_crops, propose_regions
from research_evaluation import render_research_evaluation
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
        "evaluation_rows": [],
        "review_start_time": None,
        "review_completion_time": None,
        "experiment_id": f"EXP-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "reviewer_id": "",
        "active_page": "Overview",
        "review_widget_values": {},
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def persist_review_value(key: str) -> None:
    """Copy widget state into storage that survives conditional page rendering."""
    if key in st.session_state:
        st.session_state.review_widget_values[key] = st.session_state[key]


def review_default(key: str, default):
    return st.session_state.review_widget_values.get(key, default)


def clear_review_state() -> None:
    prefixes=("decision_","label_","custom_label_","notes_","mask_choice_","morph_","small_","invert_","x1_","y1_","x2_","y2_")
    for key in list(st.session_state):
        if key.startswith(prefixes):
            del st.session_state[key]
    st.session_state.review_widget_values={}


def run_analysis(uploaded_file, settings: dict[str, int | bool]) -> None:
    clear_review_state()
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
        border_margin=float(settings["border_margin"]),
        ablation=AblationConfig(
            edge_features=bool(settings["use_edges"]), texture_features=bool(settings["use_texture"]),
            colour_features=bool(settings["use_colour"]), entropy_features=bool(settings["use_entropy"]),
            stability=bool(settings["use_stability"]), contextual_contrast=bool(settings["use_context"]),
            multi_scale_fusion=bool(settings["use_multiscale"]), region_merging=bool(settings["use_merging"]),
            mask_refinement=bool(settings["use_refinement"]),
        ),
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
    st.session_state.review_start_time = datetime.now().isoformat(timespec="seconds")
    st.session_state.review_completion_time = None
    st.session_state.preprocess_settings = settings
    ablation_path=OUTPUT_DIR/"ablation_results.csv"
    ablation_row={**settings,**proposal_result.diagnostics.to_dict(),"image":uploaded_file.name}
    existing=pd.read_csv(ablation_path) if ablation_path.exists() else pd.DataFrame()
    pd.concat([existing,pd.DataFrame([ablation_row])],ignore_index=True).to_csv(ablation_path,index=False)


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
            "border_margin": st.slider("Border exclusion margin", 0.0, 0.10, 0.025, 0.005),
            "max_regions": st.slider("Maximum final regions", 3, 20, 8, 1),
            "yolo_confidence": st.slider("YOLO confidence", 0.10, 0.90, 0.35, 0.05),
        }
        with st.expander("Ablation switches"):
            settings.update({
                "use_edges": st.checkbox("Edge features",True), "use_texture": st.checkbox("Texture features",True),
                "use_colour": st.checkbox("Colour features",True), "use_entropy": st.checkbox("Entropy features",True),
                "use_stability": st.checkbox("Stability",True), "use_context": st.checkbox("Contextual contrast",True),
                "use_multiscale": st.checkbox("Multi-scale fusion",True), "use_merging": st.checkbox("Region merging",True),
                "use_refinement": st.checkbox("Mask refinement",True),
            })

        analyze = st.button("Analyze Selected Image", type="primary", use_container_width=True)

    if analyze:
        if not uploaded_files:
            st.error("Upload at least one image before analysis.")
        else:
            with st.spinner("Extracting features and proposing candidate regions..."):
                run_analysis(uploaded_files[0], settings)

    pages = (
        "Overview",
        "Image Analysis",
        "Feature Maps",
        "Region Proposals",
        "Human Review / Labeling",
        "Dataset Export",
        "Report Generation",
        "Future Model Training",
        "Research Evaluation",
    )
    active_page = st.radio(
        "Workspace page",
        pages,
        key="active_page",
        horizontal=True,
        label_visibility="collapsed",
    )

    if active_page == "Overview":
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

    if active_page == "Image Analysis":
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

    if active_page == "Feature Maps":
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

    if active_page == "Region Proposals":
        st.subheader("Region Proposals")
        proposal_result = st.session_state.proposal_result
        if proposal_result is None:
            st.info("Run analysis to create region proposals.")
        else:
            visualization_mode = st.radio("Proposal visualization", ["boxes only", "masks only", "boxes + masks"], horizontal=True)
            st.image(proposal_result.visualization_paths[visualization_mode].as_posix(), caption="Final ranked visual anomaly candidates", use_container_width=True)
            diagnostics = proposal_result.diagnostics
            stage_counts = [
                ("Raw", diagnostics.raw_components), ("Filtered", diagnostics.after_filtering),
                ("Split", diagnostics.after_splitting), ("Merged", diagnostics.after_merging),
                ("NMS", diagnostics.after_overlap_suppression), ("Ranked", diagnostics.ranked_count),
                ("Top-K", diagnostics.final_count),
            ]
            for start in range(0, len(stage_counts), 4):
                cols = st.columns(min(4, len(stage_counts) - start))
                for col, (label, value) in zip(cols, stage_counts[start:start + 4]):
                    col.metric(label, value)
            st.dataframe(pd.DataFrame([proposal.to_row() for proposal in proposal_result.proposals]), use_container_width=True)
            show_debug = st.checkbox("Display rejected/noisy candidate stages", value=False)
            if show_debug:
                st.caption("Pipeline debug panel")
                removed = st.columns(5)
                removed[0].metric("Area removed", diagnostics.removed_by_area)
                removed[1].metric("Border removed", diagnostics.removed_by_border)
                removed[2].metric("Overlap removed", diagnostics.removed_by_overlap)
                removed[3].metric("Split operations", diagnostics.split_operations)
                removed[4].metric("Merged", diagnostics.merged_candidates)
                st.json(diagnostics.rejection_reasons)
                for stage, path in diagnostics.stage_overlay_paths.items():
                    st.image(path.as_posix(), caption=f"{stage.replace('_', ' ').title()}", use_container_width=True)
            st.caption("Algorithm comparison")
            comparison_cols = st.columns(4)
            for col, (name, path) in zip(comparison_cols, proposal_result.comparison_paths.items()):
                col.image(path.as_posix(), caption=f"{name} | {proposal_result.comparison_counts[name]} regions", use_container_width=True)
            for proposal in proposal_result.proposals:
                with st.expander(f"{proposal.region_id} technical evidence"):
                    mask_cols=st.columns(3)
                    mask_cols[0].image(proposal.raw_mask_path.as_posix(),caption="Raw mask",clamp=True)
                    mask_cols[1].image(proposal.mask_path.as_posix(),caption="Refined mask",clamp=True)
                    mask_cols[2].image(proposal.context_mask_path.as_posix(),caption="Local context ring",clamp=True)
                    score_cols=st.columns(4)
                    score_cols[0].metric("Anomaly evidence",f"{proposal.anomaly_evidence_score:.1f}")
                    score_cols[1].metric("Mask reliability",f"{proposal.mask_reliability_score:.1f}")
                    score_cols[2].metric("Review priority",f"{proposal.priority.score:.1f}")
                    score_cols[3].metric("Coherence",f"{proposal.coherence_score:.2f}")
                    contribution=pd.DataFrame({"Feature":list(proposal.feature_contributions),"Contribution (%)":list(proposal.feature_contributions.values())})
                    st.bar_chart(contribution.set_index("Feature"))
                    st.caption(f"Border penalty {proposal.border_penalty:.3f} | area reduction {proposal.area_reduction*100:.1f}% | boundary smoothness {proposal.boundary_smoothness:.3f}")
            st.download_button(
                "Download Combined Binary Mask",
                proposal_result.combined_mask_path.read_bytes(),
                file_name=proposal_result.combined_mask_path.name,
                mime="image/png",
            )

    if active_page == "Human Review / Labeling":
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
                    decision_key=f"decision_{proposal.region_id}"
                    decision_options=["uncertain","accept","reject"]
                    decision_default=review_default(decision_key,"uncertain")
                    decision = st.radio("Review decision",decision_options,index=decision_options.index(decision_default),horizontal=True,
                        key=decision_key,on_change=persist_review_value,args=(decision_key,))
                    label_key=f"label_{proposal.region_id}"
                    label_default=review_default(label_key,"unassigned")
                    label = st.selectbox("Candidate label",DEFAULT_LABEL_CLASSES,index=DEFAULT_LABEL_CLASSES.index(label_default),
                        key=label_key,on_change=persist_review_value,args=(label_key,))
                    custom_key=f"custom_label_{proposal.region_id}"
                    custom = st.text_input("Optional custom label",value=review_default(custom_key,""),key=custom_key,
                        on_change=persist_review_value,args=(custom_key,))
                    notes_key=f"notes_{proposal.region_id}"
                    notes = st.text_area("Notes",value=review_default(notes_key,""),key=notes_key,height=70,
                        on_change=persist_review_value,args=(notes_key,))
                    st.caption("Manual mask correction")
                    x1,y1,x2,y2=proposal.bbox; bbox_cols=st.columns(4)
                    bbox_keys=[f"x1_{proposal.region_id}",f"y1_{proposal.region_id}",f"x2_{proposal.region_id}",f"y2_{proposal.region_id}"]
                    bbox_defaults=[x1,y1,x2,y2]; bbox_limits=[(0,st.session_state.processed.shape[1]-1),(0,st.session_state.processed.shape[0]-1),(1,st.session_state.processed.shape[1]),(1,st.session_state.processed.shape[0])]
                    bbox_values=[]
                    for col,caption,key,default,limits in zip(bbox_cols,["x1","y1","x2","y2"],bbox_keys,bbox_defaults,bbox_limits):
                        bbox_values.append(int(col.number_input(caption,limits[0],limits[1],int(review_default(key,default)),key=key,
                            on_change=persist_review_value,args=(key,))))
                    corrected_bbox=tuple(bbox_values)
                    correction_cols=st.columns(4)
                    mask_key=f"mask_choice_{proposal.region_id}"; mask_options=["refined","raw"]
                    mask_source=correction_cols[0].selectbox("Mask source",mask_options,index=mask_options.index(review_default(mask_key,"refined")),key=mask_key,
                        on_change=persist_review_value,args=(mask_key,))
                    morph_key=f"morph_{proposal.region_id}"
                    morphology=correction_cols[1].slider("Erode / dilate",-4,4,int(review_default(morph_key,0)),key=morph_key,
                        on_change=persist_review_value,args=(morph_key,))
                    small_key=f"small_{proposal.region_id}"
                    remove_small=correction_cols[2].number_input("Remove below",0,5000,int(review_default(small_key,0)),25,key=small_key,
                        on_change=persist_review_value,args=(small_key,))
                    invert_key=f"invert_{proposal.region_id}"
                    invert=correction_cols[3].checkbox("Invert",bool(review_default(invert_key,False)),key=invert_key,
                        on_change=persist_review_value,args=(invert_key,))
                    corrected_path,corrected_metrics=correct_region_mask(proposal,corrected_bbox,mask_source,morphology,int(remove_small),invert,st.session_state.image_path.stem)
                    st.image(corrected_path.as_posix(),caption="Corrected annotation mask",clamp=True)
                    st.json(corrected_metrics)
                    final_label = custom.strip() if custom.strip() else label
                    corrected=corrected_bbox!=proposal.bbox or mask_source!="refined" or morphology!=0 or remove_small!=0 or invert
                    annotations.append(build_annotation(st.session_state.image_name,proposal,decision=="accept",final_label,notes,
                        decision=decision,corrected_bbox=corrected_bbox,corrected_mask_path=str(corrected_path),mask_source="corrected" if corrected else "refined"))
            if st.button("Save Review Metadata", type="primary"):
                st.session_state.annotations = annotations
                st.session_state.review_completion_time = datetime.now().isoformat(timespec="seconds")
                st.success(f"Saved {len(annotations)} reviewed region records in session state.")

    if active_page == "Dataset Export":
        st.subheader("Dataset Export")
        if st.session_state.image_path is None:
            st.info("Analyze and review an image before exporting.")
        elif not st.session_state.annotations:
            st.warning("Save review metadata first. Export uses accepted regions and their candidate labels.")
        else:
            accepted = [ann for ann in st.session_state.annotations if ann.accepted and ann.label not in {"ignore","unassigned"}]
            st.metric("Accepted candidate regions", len(accepted))
            if st.button("Export Reviewed Dataset Files", type="primary"):
                try:
                    paths = export_dataset(st.session_state.image_path,st.session_state.processed.shape[:2],st.session_state.annotations)
                    st.success("Dataset export completed.")
                    st.json({name: path.as_posix() for name, path in paths.items()})
                except ValueError as error:
                    st.error(str(error))
            st.caption("Annotation evaluation")
            if st.button("Evaluate Proposal Methods"):
                references=[ann.bbox for ann in st.session_state.annotations if ann.accepted and ann.label not in {"unassigned","ignore"}]
                reference_masks=[]
                for ann in st.session_state.annotations:
                    if ann.accepted and ann.label not in {"unassigned","ignore"}:
                        mask=cv2.imread(ann.mask_path,cv2.IMREAD_GRAYSCALE)
                        if mask is not None: reference_masks.append(mask)
                if not references:
                    st.error("Accept and intentionally label at least one reviewed region before evaluation.")
                else:
                    fm=st.session_state.feature_maps; result=st.session_state.proposal_result
                    raw_union=np.zeros(st.session_state.processed.shape[:2],np.uint8)
                    for proposal in result.proposals:
                        raw=cv2.imread(str(proposal.raw_mask_path),0)
                        if raw is not None: raw_union=cv2.bitwise_or(raw_union,raw)
                    methods={"contour-only":fm.contour_map,"fixed-threshold":((fm.anomaly_strength>128).astype(np.uint8)*255),
                             "multi-scale-fused":raw_union,"refined-contextual":cv2.imread(str(result.combined_mask_path),0)}
                    rows=[]
                    for method,mask in methods.items():
                        boxes=[item.bbox for item in _components(mask)]
                        rows.append(evaluate_method(boxes,references,[mask],reference_masks,st.session_state.annotations,
                            image_name=st.session_state.image_name,method=method))
                    st.session_state.evaluation_rows=rows
            if st.session_state.evaluation_rows:
                per_image,dataset=evaluation_tables(st.session_state.evaluation_rows)
                st.dataframe(per_image,use_container_width=True); st.dataframe(dataset,use_container_width=True)
                st.download_button("Download Evaluation CSV",per_image.to_csv(index=False).encode(),"proposal_evaluation.csv","text/csv")

    if active_page == "Report Generation":
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

    if active_page == "Future Model Training":
        st.subheader("Future Model Training")
        st.write(
            "After reviewing and exporting enough labeled candidate regions, train YOLO detection or segmentation with Ultralytics. "
            "The app will automatically show trained inference separately when `models/best.pt` exists."
        )
        st.code("python train.py --data datasets/data.yaml --task detect --model yolo11n.pt --epochs 80 --imgsz 640", language="bash")
        st.code("python train.py --data datasets/data.yaml --task segment --model yolo11n-seg.pt --epochs 80 --imgsz 640", language="bash")
        st.info("SAM/SAM2 integration is future-ready: use proposed boxes as prompts, then replace rectangular masks with refined masks.")

    if active_page == "Research Evaluation":
        render_research_evaluation(
            output_dir=OUTPUT_DIR,
            image_name=st.session_state.image_name,
            annotations=st.session_state.annotations,
            proposal_result=st.session_state.proposal_result,
            feature_maps=st.session_state.feature_maps,
            review_start_time=st.session_state.review_start_time,
            review_completion_time=st.session_state.review_completion_time,
        )


if __name__ == "__main__":
    main()

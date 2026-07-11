"""Streamlit page for research dataset intake and validation."""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from research_dataset import (
    ANNOTATION_FORMATS, GROUND_TRUTH_OPTIONS, SOURCE_TYPES, DatasetMetadata,
    DatasetRegistry, extract_zip, ingest_files, licence_allows_public_export,
    prepare_split, register_synthetic_benchmark,
)
from dataset_management import DatasetManager


def render_dataset_intake(base_dir:Path)->None:
    st.subheader("Research Dataset Intake")
    st.caption("Register provenance, validate files, prepare leakage-safe splits, and export reproducible manifests before experiments.")
    registry=DatasetRegistry(base_dir/"research_data")
    metadata=_metadata_form()
    _import_section(registry,metadata)
    st.divider(); _dashboard(registry)
    st.divider(); _manage_datasets(registry)
    st.divider(); _split_section(registry)
    st.divider(); _ground_truth_review(registry)


def _metadata_form():
    st.markdown("### Dataset Registration")
    row=st.columns(3)
    dataset_id=row[0].text_input("Dataset ID")
    dataset_name=row[1].text_input("Dataset name")
    version=row[2].text_input("Dataset version",value="1.0")
    row=st.columns(3)
    source_type=row[0].selectbox("Source type",SOURCE_TYPES)
    source_name=row[1].text_input("Source name")
    source_reference=row[2].text_input("Source URL or reference")
    row=st.columns(3)
    provider=row[0].text_input("Provider/author")
    licence=row[1].text_input("Licence",value="unknown")
    acquired=row[2].date_input("Date acquired",value=date.today())
    flags=st.columns(2)
    redistribution=flags[0].checkbox("Redistribution allowed")
    commercial=flags[1].checkbox("Commercial use allowed")
    citation=st.text_area("Citation text")
    row=st.columns(3)
    domain=row[0].text_input("Domain/category",value="structural inspection")
    ground=row[1].selectbox("Ground-truth status",GROUND_TRUTH_OPTIONS)
    annotation=row[2].selectbox("Annotation format",ANNOTATION_FORMATS)
    notes=st.text_area("Dataset notes")
    return DatasetMetadata(dataset_id,dataset_name,version,source_type,source_name,source_reference,provider,licence,redistribution,commercial,citation,acquired.isoformat(),domain,ground,annotation,notes)


def _import_section(registry,metadata):
    st.markdown("### Import Files")
    mode=st.radio("Import mode",["Single image","Batch images","ZIP dataset"],horizontal=True)
    files=[]; zip_annotations={}
    if mode=="Single image":
        upload=st.file_uploader("Single-image import",type=["jpg","jpeg","png","bmp","tif","tiff","webp"],key="intake_single")
        if upload: files=[(upload.name,upload.getvalue())]
    elif mode=="Batch images":
        uploads=st.file_uploader("Batch-image import",type=["jpg","jpeg","png","bmp","tif","tiff","webp"],accept_multiple_files=True,key="intake_batch")
        files=[(item.name,item.getvalue()) for item in uploads]
    else:
        upload=st.file_uploader("ZIP dataset import",type=["zip"],key="intake_zip")
        if upload:
            try: files,zip_annotations=extract_zip(upload.getvalue())
            except Exception as error: st.error(f"Invalid ZIP: {error}")
    annotation_uploads=st.file_uploader("Optional annotation import",accept_multiple_files=True,key="intake_annotations")
    annotations={item.name:item.getvalue() for item in annotation_uploads}; annotations.update(zip_annotations)
    registration_mode=st.radio("Existing Dataset ID/version",["cancel","replace","create a new version"],horizontal=True)
    if st.button("Register and Import Dataset",type="primary"):
        try:
            if not files: raise ValueError("Select at least one image")
            if registration_mode=="create a new version": raise ValueError("Enter a new Dataset version before importing")
            records,report=ingest_files(registry,metadata,files,annotations,registration_mode)
            st.success(f"Imported {len(records)} manifest rows.")
            st.json(report.to_dict())
            report_dir=registry.root/"reports"/metadata.dataset_id
            st.download_button("Download validation CSV",(report_dir/"validation_records.csv").read_bytes(),"validation_records.csv","text/csv")
            st.download_button("Download validation JSON",(report_dir/"validation_report.json").read_bytes(),"validation_report.json","application/json")
        except ValueError as error: st.error(str(error))

    st.caption("Controlled synthetic benchmark")
    seed=int(st.number_input("Synthetic random seed",0,1000000,11))
    replace=st.checkbox("Replace existing synthetic dataset")
    if st.button("Generate and Register Synthetic Benchmark"):
        try:
            records,report=register_synthetic_benchmark(registry,seed=seed,registration_mode="replace" if replace else "cancel")
            st.success(f"Registered {len(records)} synthetic images and exact masks."); st.json(report.to_dict())
        except ValueError as error: st.error(str(error))


def _dashboard(registry):
    st.markdown("### Dataset Dashboard")
    datasets=registry.datasets(); images=registry.images()
    if datasets.empty:
        st.info("No registered datasets are available."); return
    summary=[]
    for _,dataset in datasets.iterrows():
        subset=images[images.dataset_id==dataset.dataset_id] if not images.empty else images
        exact=int((subset.duplicate_type=="exact image duplicate").sum()) if len(subset) else 0; near=int((subset.duplicate_type=="near duplicate").sum()) if len(subset) else 0
        summary.append({"dataset_id":dataset.dataset_id,"version":dataset.dataset_version,"images":len(subset),"unique_images":len(subset)-exact,"split_eligible":int(subset.split_eligible.astype(int).sum()) if len(subset) else 0,"annotation_coverage":float(subset.annotation_path.astype(bool).mean()) if len(subset) else 0,"ground_truth":dataset.ground_truth_status,"licence":dataset.licence,"exact_duplicates":exact,"near_duplicates":near,"duplicate_groups":int(subset[subset.duplicate_group_id!=""].duplicate_group_id.nunique()) if len(subset) else 0,"validation_warnings":int((subset.corruption_status!="valid").sum()) if len(subset) else 0})
    table=pd.DataFrame(summary); st.dataframe(table,width="stretch",hide_index=True)
    _compact_bar(table,"dataset_id","images","Image Counts")
    _compact_bar(table,"dataset_id","annotation_coverage","Annotation Coverage",percent=True)
    if not images.empty:
        split=images.groupby("split").size().reset_index(name="images"); _compact_bar(split,"split","images","Split Distribution")
        classes=images[images.primary_class!=""].groupby("primary_class").size().reset_index(name="images"); _compact_bar(classes,"primary_class","images","Class Distribution")
    manifest=registry.registry_dir/"dataset_manifest.json"
    if manifest.exists(): st.download_button("Download Dataset Manifest JSON",manifest.read_bytes(),"dataset_manifest.json","application/json")
    selected=st.selectbox("Public export licence check",datasets.dataset_id.tolist())
    override=st.checkbox("Explicitly override unknown/restricted licence warning")
    allowed,message=licence_allows_public_export(registry.metadata(selected),override); (st.success if allowed else st.warning)(message)

    details=images[images.duplicate_type!="unique"]
    st.markdown("#### Duplicate Details")
    if details.empty: st.info("No exact or near image duplicates detected.")
    else:
        columns=["image_id","original_filename","sha256_hash","duplicate_group_id","canonical_image_id","duplicate_type","generation_seed","synthetic_anomaly_type","generation_parameters"]
        st.dataframe(details[columns],width="stretch",hide_index=True)


def _manage_datasets(registry):
    st.markdown("### Manage Research Datasets")
    datasets=registry.datasets(); images=registry.images(); manager=DatasetManager(registry)
    if datasets.empty: st.info("No registered datasets are available."); return
    table=[]
    for _,item in datasets.iterrows():
        subset=images[(images.dataset_id==item.dataset_id)&(images.dataset_version==item.dataset_version)]
        table.append({"dataset_id":item.dataset_id,"dataset_name":item.dataset_name,"dataset_version":item.dataset_version,"source_type":item.source_type,"image_count":len(subset),"annotation_count":int(subset.annotation_path.astype(bool).sum()),"exact_duplicate_count":int((subset.duplicate_type=="exact image duplicate").sum()),"near_duplicate_count":int((subset.duplicate_type=="near duplicate").sum()),"split_status":", ".join(sorted(subset.split.unique())),"created_at":item.registered_timestamp,"storage_path":str(registry.root/"raw"/item.dataset_id),"licence":item.licence,"ground_truth_status":item.ground_truth_status})
    frame=pd.DataFrame(table); filters=st.columns(5)
    dataset_filter=filters[0].selectbox("Dataset ID",["All"]+sorted(frame.dataset_id.unique().tolist()),key="manage_dataset_filter")
    name_filter=filters[1].selectbox("Dataset name",["All"]+sorted(frame.dataset_name.unique().tolist()))
    version_filter=filters[2].selectbox("Version",["All"]+sorted(frame.dataset_version.unique().tolist()))
    source_filter=filters[3].selectbox("Source type",["All"]+sorted(frame.source_type.unique().tolist()))
    date_range=filters[4].date_input("Date range",value=[])
    filtered=frame.copy()
    for column,value in (("dataset_id",dataset_filter),("dataset_name",name_filter),("dataset_version",version_filter),("source_type",source_filter)):
        if value!="All": filtered=filtered[filtered[column]==value]
    if len(date_range)==2: filtered=filtered[pd.to_datetime(filtered.created_at).dt.date.between(date_range[0],date_range[1])]
    st.dataframe(filtered,width="stretch",hide_index=True)
    selected=st.selectbox("Selected dataset",frame.dataset_id.unique(),key="managed_dataset"); versions=frame[frame.dataset_id==selected].dataset_version.tolist(); version=st.selectbox("Selected version",versions)
    actions=["Delete selected dataset version","Delete every version of selected Dataset ID","Delete only generated split files","Delete only validation reports","Delete only derived/processed files","Delete only synthetic benchmark datasets","Clear all Development/Test datasets","Reset the full Research Dataset Intake store"]
    action=st.selectbox("Management action",actions); mode=st.radio("Deletion mode",["Metadata only","Generated files only","Complete deletion"],horizontal=True)
    preview_mode="metadata" if mode=="Metadata only" else ("generated" if mode=="Generated files only" else "complete")
    preview_mode={"Delete only generated split files":"splits","Delete only validation reports":"reports","Delete only derived/processed files":"processed"}.get(action,preview_mode)
    if action=="Delete only synthetic benchmark datasets" and frame.loc[frame.dataset_id==selected,"source_type"].iloc[0]!="synthetic": st.warning("Select a synthetic dataset for this action.")
    target_version=version if action=="Delete selected dataset version" else None
    preview=manager.preview(selected,target_version,preview_mode)
    st.markdown("#### Cleanup Preview"); st.json({"registry database rows":preview.database_rows,"manifest records":preview.database_rows,"raw image files / processed images / annotation files / masks / split manifests / reports":len(preview.files),"files to remove":preview.files,"total disk size":preview.total_bytes,"linked experiments":preview.linked_experiments,"affected manifests":preview.affected_manifests,"generated chart/dashboard data":"recomputed immediately"})
    reviewer=st.text_input("Deletion reviewer/user"); confirm=st.checkbox("I confirm this destructive dataset action"); typed=st.text_input("Type the exact Dataset ID to confirm")
    unlink=st.checkbox("Unlink development experiments while preserving audit metadata"); cascade=st.checkbox("Cascade delete linked development experiments")
    label="Delete `synthetic-controlled`" if selected=="synthetic-controlled" else "Final delete button"
    if st.button(label,disabled=not(confirm and typed==selected)):
        try: st.success(manager.delete(selected,target_version,preview_mode,reviewer,unlink,cascade)); st.rerun()
        except ValueError as error: st.error(str(error))
    if selected=="synthetic-controlled" and st.button("Regenerate Synthetic Benchmark"):
        try: records,_=register_synthetic_benchmark(registry,version=version,seed=11,registration_mode="replace"); st.success(f"Regenerated {len(records)} images."); st.rerun()
        except ValueError as error: st.error(str(error))
    st.caption("Deleted files first move to research_data/.trash/<timestamp>/<dataset_id>/")
    audits=list((registry.root/".trash").glob("*/*/deletion_audit.json"))
    if audits:
        audit=st.selectbox("Trash audit",audits,format_func=lambda path:str(path.relative_to(registry.root)))
        if st.button("Restore deleted dataset"): st.success(f"Restored {manager.restore(audit)} files."); st.rerun()
        empty_confirm=st.checkbox("Confirm permanent dataset trash removal")
        if st.button("Permanently empty dataset trash",disabled=not empty_confirm): st.success(f"Removed {manager.empty_trash()} trash files."); st.rerun()
    dataset_images=images[images.dataset_id==selected]
    if not dataset_images.empty:
        item_id=st.selectbox("Dataset image item",dataset_images.image_id,format_func=lambda value:dataset_images.loc[dataset_images.image_id==value,"original_filename"].iloc[0]); eligible=bool(dataset_images.loc[dataset_images.image_id==item_id,"split_eligible"].iloc[0])
        if st.button("Exclude selected duplicate from split eligibility" if eligible else "Restore excluded image"): manager.set_split_eligible(item_id,not eligible); st.rerun()
        physical=st.checkbox("Confirm physical file deletion")
        if st.button("Delete selected image from dataset",disabled=not physical): manager.delete_item(item_id,delete_file=True); st.rerun()
        if st.button("Delete selected annotation",disabled=not physical): manager.delete_item(item_id,delete_annotation=True); st.rerun()


def _split_section(registry):
    st.markdown("### Train / Validation / Test Split")
    datasets=registry.datasets()
    if datasets.empty: st.info("Register a dataset before preparing splits."); return
    dataset_id=st.selectbox("Dataset for split",datasets.dataset_id.unique(),key="split_dataset")
    row=st.columns(4)
    train=int(row[0].number_input("Train %",0,100,70)); validation=int(row[1].number_input("Validation %",0,100,15)); test=int(row[2].number_input("Test %",0,100,15)); seed=int(row[3].number_input("Split seed",0,1000000,42))
    images=registry.images(dataset_id); total=len(images[images.corruption_status=="valid"])
    st.json({"preview_train":round(total*train/100),"preview_validation":round(total*validation/100),"preview_test":total-round(total*train/100)-round(total*validation/100),"deterministic_seed":seed})
    override=st.checkbox("Override detected split leakage")
    confirm=st.checkbox("Confirm split finalization")
    if st.button("Finalize Split",disabled=not confirm):
        try:
            result,leaks,path=prepare_split(registry,dataset_id,(train/100,validation/100,test/100),seed,override)
            st.success(f"Saved split manifest: {path.name}"); st.json({"split_counts":result.groupby('split').size().to_dict(),"leakage_checks":leaks})
        except ValueError as error: st.error(str(error))


def _ground_truth_review(registry):
    st.markdown("### Reference Ground-Truth Review")
    images=registry.images()
    if images.empty: st.info("Import images before adding reference reviews."); return
    image_id=st.selectbox("Image ID",images.image_id.tolist(),format_func=lambda value:f"{value[:8]} — {images.loc[images.image_id==value,'original_filename'].iloc[0]}")
    outcome=st.radio("Reference outcome",["uncertain","anomaly present","no anomaly"],horizontal=True)
    row=st.columns(2); reviewer=row[0].text_input("Reference reviewer ID"); confidence=row[1].slider("Confidence",0.0,1.0,.5,.05)
    bbox_cols=st.columns(4); bbox=[int(col.number_input(label,0,100000,0,key=f"gt_{label}")) for col,label in zip(bbox_cols,["x1","y1","x2","y2"])]
    mask=st.file_uploader("Upload reference mask",type=["png","tif","tiff"],key="gt_mask")
    notes=st.text_area("Reference notes")
    if st.button("Save Reference Ground Truth"):
        try: registry.save_review(image_id,outcome,reviewer,confidence,notes,bbox if bbox[2]>bbox[0] and bbox[3]>bbox[1] else None,mask.getvalue() if mask else None); st.success("Saved reference ground truth separately from proposals and predictions.")
        except ValueError as error: st.error(str(error))


def _compact_bar(frame,x,y,title,percent=False):
    valid=frame[[x,y]].dropna() if not frame.empty else frame
    if valid.empty: st.info(f"{title}: no valid data available."); return
    height=2.6 if len(valid)==1 else (3.4 if len(valid)<=4 else 4.4)
    fig,axis=plt.subplots(figsize=(8,height)); bars=axis.bar(valid[x].astype(str),valid[y]); axis.set_title(title); axis.set_xlabel(x.replace('_',' ').title()); axis.set_ylabel(y.replace('_',' ').title()); axis.tick_params(axis="x",rotation=18)
    if percent: axis.set_ylim(0,1)
    for bar,value in zip(bars,valid[y]): axis.text(bar.get_x()+bar.get_width()/2,bar.get_height(),f"{value:.1%}" if percent else f"{value:g}",ha="center",va="bottom",fontsize=8)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

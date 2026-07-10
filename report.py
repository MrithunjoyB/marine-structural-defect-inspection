"""Professional PDF reports for StructVision-AI analyses."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from labeling import ReviewedAnnotation
from region_proposal import ProposalResult
from yolo_inference import YoloInferenceResult


LIMITATIONS = (
    "Before a trained model is available, candidate regions are generated from classical computer vision feature maps. "
    "They indicate visually significant areas for review and dataset creation, not certified defect classifications."
)


def generate_pdf_report(
    report_dir: Path,
    image_name: str,
    preprocessing_settings: dict[str, object],
    feature_paths: dict[str, Path],
    proposal_result: ProposalResult,
    annotations: list[ReviewedAnnotation],
    yolo_result: YoloInferenceResult | None,
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"structvision_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}.pdf"
    doc = SimpleDocTemplate(str(report_path), pagesize=A4, rightMargin=34, leftMargin=34, topMargin=34, bottomMargin=34)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("StructVision-AI", styles["Title"]),
        Paragraph("Foundation-Model-Assisted Visual Inspection and Dataset Generation Report", styles["Heading2"]),
        Paragraph(f"<b>Image filename:</b> {image_name}", styles["BodyText"]),
        Paragraph(f"<b>Analysis date/time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["BodyText"]),
        Paragraph(f"<b>YOLO status:</b> {yolo_result.message if yolo_result else 'Not evaluated'}", styles["BodyText"]),
        Spacer(1, 0.12 * inch),
    ]

    story.append(Paragraph("Preprocessing Settings", styles["Heading2"]))
    settings_text = ", ".join(f"{key}: {value}" for key, value in preprocessing_settings.items())
    story.append(Paragraph(settings_text or "Default preprocessing settings used.", styles["BodyText"]))
    story.append(Spacer(1, 0.1 * inch))

    if proposal_result.overlay_path.exists():
        story.append(Paragraph("Highlighted Region Proposal Image", styles["Heading2"]))
        story.append(Image(str(proposal_result.overlay_path), width=6.3 * inch, height=3.7 * inch, kind="proportional"))
        story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("Feature Map Thumbnails", styles["Heading2"]))
    feature_images = [path for path in feature_paths.values() if path.exists()][:4]
    if feature_images:
        row = [Image(str(path), width=1.45 * inch, height=1.0 * inch, kind="proportional") for path in feature_images]
        story.append(Table([row]))
    else:
        story.append(Paragraph("Feature map files were not available for this report.", styles["BodyText"]))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("Region Summary", styles["Heading2"]))
    rows = [["ID", "BBox", "Area", "Edge", "Texture", "Color", "Score", "Priority"]]
    for proposal in proposal_result.proposals[:24]:
        row = proposal.to_row()
        rows.append(
            [
                row["Region ID"],
                row["BBox"],
                row["Pixel Area"],
                row["Edge Density"],
                row["Texture Score"],
                row["Color Variation"],
                row["Priority Score"],
                row["Priority Label"],
            ]
        )
    if len(rows) == 1:
        rows.append(["No regions", "-", "-", "-", "-", "-", "-", "-"])
    table = Table(rows, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#203864")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(table)

    if annotations:
        story.append(Spacer(1, 0.1 * inch))
        story.append(Paragraph("Human Review Labels", styles["Heading2"]))
        label_rows = [["Region", "Accepted", "Label", "Priority"]]
        for ann in annotations[:28]:
            label_rows.append([ann.region_id, str(ann.accepted), ann.label, ann.priority_label])
        label_table = Table(label_rows, repeatRows=1)
        label_table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.35, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 8)]))
        story.append(label_table)

    story.extend(
        [
            Spacer(1, 0.15 * inch),
            Paragraph("Limitations", styles["Heading2"]),
            Paragraph(LIMITATIONS, styles["BodyText"]),
            Paragraph(
                "Future training note: export reviewed annotations to YOLO detection or segmentation format, train with Ultralytics, "
                "then place the resulting model at models/best.pt for separate trained inference.",
                styles["BodyText"],
            ),
        ]
    )
    doc.build(story)
    return report_path

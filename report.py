"""PDF report generation for inspection results."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from detect import DetectionResult
from severity import SeverityResult


DISCLAIMER = "This is an AI-assisted visual inspection prototype and should not replace certified marine or structural inspection."


def generate_pdf_report(
    report_dir: Path,
    image_name: str,
    detection: DetectionResult,
    severity: SeverityResult,
    interpretation: str,
    actions: list[str],
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"inspection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}.pdf"

    doc = SimpleDocTemplate(str(report_path), pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    heading = styles["Heading2"]
    body = styles["BodyText"]
    small = ParagraphStyle("small", parent=body, fontSize=8, leading=10)

    story = [
        Paragraph("AI-Based Visual Inspection and Defect Severity Analysis", title_style),
        Paragraph("Marine Structural Components Inspection Report", heading),
        Paragraph(f"<b>Date and time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body),
        Paragraph(f"<b>Uploaded image:</b> {image_name}", body),
        Paragraph(f"<b>Detection mode:</b> {detection.mode}", body),
        Spacer(1, 0.15 * inch),
    ]

    if detection.annotated_image_path.exists():
        story.append(Image(str(detection.annotated_image_path), width=6.0 * inch, height=3.6 * inch, kind="proportional"))
        story.append(Spacer(1, 0.15 * inch))

    summary_rows = [["Defect Type", "Confidence", "BBox", "Area px", "Rel. Area %"]]
    for defect in detection.defects:
        row = defect.to_table_row()
        summary_rows.append(
            [
                row["Defect Type"],
                row["Confidence"],
                row["Bounding Box"],
                row["Area (px)"],
                row["Relative Area (%)"],
            ]
        )
    if len(summary_rows) == 1:
        summary_rows.append(["No visible defect detected", "-", "-", "-", "-"])

    table = Table(summary_rows, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.extend(
        [
            Paragraph("Defect Summary", heading),
            table,
            Spacer(1, 0.15 * inch),
            Paragraph("Severity Estimate", heading),
            Paragraph(f"<b>Score:</b> {severity.score}/100 &nbsp;&nbsp; <b>Level:</b> {severity.label}", body),
            Paragraph("Engineering Interpretation", heading),
            Paragraph(interpretation, body),
            Paragraph("Recommended Inspection Action", heading),
        ]
    )

    for action in actions:
        story.append(Paragraph(f"- {action}", body))

    story.extend([Spacer(1, 0.15 * inch), Paragraph(f"<b>Disclaimer:</b> {DISCLAIMER}", small)])
    doc.build(story)
    return report_path

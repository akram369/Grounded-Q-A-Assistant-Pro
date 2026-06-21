"""Build the included multi-page PDF knowledge-base document."""

from pathlib import Path

from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_documents" / "orbital_debris_management.txt"
OUTPUT = ROOT / "data" / "orbital_debris_management.pdf"


def footer(canvas, document):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(letter[0] / 2, 0.42 * inch, f"Orbital Debris Management | Page {document.page}")
    canvas.restoreState()


def build() -> None:
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    title, *body = lines
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleCustom", parent=styles["Title"], alignment=TA_CENTER, fontSize=20, leading=25, spaceAfter=24
    )
    heading_style = ParagraphStyle(
        "HeadingCustom", parent=styles["Heading2"], fontSize=13, leading=16, spaceBefore=12, spaceAfter=7
    )
    body_style = ParagraphStyle(
        "BodyCustom", parent=styles["BodyText"], fontSize=10.5, leading=15, spaceAfter=10
    )
    story = [Paragraph(title.title(), title_style), Spacer(1, 8)]
    paragraphs = [line.strip() for line in body if line.strip()]
    for index, paragraph in enumerate(paragraphs):
        is_heading = len(paragraph.split()) <= 6 and not paragraph.endswith(".")
        story.append(Paragraph(paragraph, heading_style if is_heading else body_style))
        if index == 8:
            story.append(PageBreak())

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(OUTPUT), pagesize=letter, rightMargin=0.75 * inch, leftMargin=0.75 * inch,
        topMargin=0.72 * inch, bottomMargin=0.7 * inch, title="Orbital Debris Management and Spaceflight Safety"
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    build()


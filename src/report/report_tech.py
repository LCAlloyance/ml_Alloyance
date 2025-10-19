import sys
import os

from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER

# ------------------------------------------------------
# Import RAG-based text generation functions
# ------------------------------------------------------
try:
    from src.rag.rag_pipline import generate_summary, generate_circularity_analysis
except ModuleNotFoundError:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAG_DIR = os.path.join(BASE_DIR, "rag")
    sys.path.append(RAG_DIR)
    from rag_pipline import generate_summary, generate_circularity_analysis


# ------------------------------------------------------
# PDF STYLE HELPERS
# ------------------------------------------------------
def create_styles():
    """Create and return a set of reusable PDF styles."""
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='CustomTitle',
        parent=styles['Title'],
        fontSize=20,
        spaceAfter=30,
        textColor=colors.HexColor('#1f4e79'),
        alignment=TA_CENTER
    ))
    styles.add(ParagraphStyle(
        name='SectionHeader',
        parent=styles['Heading2'],
        fontSize=14,
        spaceBefore=20,
        spaceAfter=10,
        textColor=colors.HexColor('#2e5c8a')
    ))
    return styles


# ------------------------------------------------------
# PDF SECTION HELPERS
# ------------------------------------------------------
def add_title_page(story, styles):
    """Create and append the title page to the report."""
    story.append(Paragraph("Sustainability & Circularity Analysis Report", styles['CustomTitle']))
    story.append(Spacer(1, 50))

    report_data = [
        ['Report Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ['Prepared By:', 'AI-Powered Multi-Agent System'],
        ['Methodology:', 'Retrieval-Augmented Generation + Multi-Agent Workflow']
    ]
    report_table = Table(report_data, colWidths=[2 * inch, 3 * inch])
    report_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8f9fa')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(report_table)
    story.append(PageBreak())


def add_section(story, styles, title, content):
    """Add a section with title and text."""
    story.append(Paragraph(title, styles['SectionHeader']))
    story.append(Paragraph(content if content else "No data available for this section.", styles['Normal']))
    story.append(Spacer(1, 12))


# ------------------------------------------------------
# MAIN REPORT GENERATOR (DICT → PDF)
# ------------------------------------------------------
def generate_report_from_dict(data: dict, output_file=None):
    """Generate a sustainability report PDF directly from a Python dictionary."""
    if not isinstance(data, dict):
        raise ValueError("❌ Input must be a dictionary.")

    # ✅ Define output directory relative to project root
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # → ml_Alloyance/
    output_dir = os.path.join(base_dir, "generatedpdf")

    # ✅ Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)

    # ✅ Default filename if not provided
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"sustainability_report_{timestamp}.pdf")

    styles = create_styles()
    story = []

    # Title Page
    add_title_page(story, styles)

    # Extract values
    material = data.get("Raw Material Type", "Unknown Material")
    circ_score = data.get("Circularity_Score", 0)
    recycled_content = data.get("Recycled Content (%)", 0)
    reuse_potential = data.get("Reuse Potential (%)", 0)
    recovery_rate = data.get("Recovery Rate (%)", 0)

    # RAG-based generation
    exec_summary = generate_summary(material, circ_score, recycled_content, reuse_potential, recovery_rate)
    circ_analysis = generate_circularity_analysis(material, circ_score, recycled_content, reuse_potential, recovery_rate)

    # Add sections
    add_section(story, styles, "Executive Summary", exec_summary)
    add_section(story, styles, "Circularity Analysis", circ_analysis)
    add_section(story, styles, "Data Quality Assessment", data.get("Data Gaps"))
    add_section(story, styles, "Hotspot Analysis", data.get("Hotspot Analysis"))
    add_section(story, styles, "Circularity Strategies", data.get("Circular Strategies"))
    add_section(story, styles, "Scenario Modeling", data.get("Scenario Modeling"))
    add_section(story, styles, "Compliance & Risk Assessment", data.get("Compliance"))

    # Disclaimer
    story.append(PageBreak())
    story.append(Paragraph("Appendices", styles['SectionHeader']))
    story.append(Paragraph(
        "This report was automatically generated using an AI-powered multi-agent RAG pipeline. "
        "Results should be validated by domain experts before being used for critical decision-making.",
        styles['Normal']
    ))

    # Build final PDF
    doc = SimpleDocTemplate(
        output_file, pagesize=A4,
        rightMargin=72, leftMargin=72,
        topMargin=72, bottomMargin=18
    )
    doc.build(story)

    return output_file


# ------------------------------------------------------
# STANDALONE TEST
# ------------------------------------------------------
if __name__ == "__main__":
    test_data = {
    "Process Stage": "Manufacturing",
    "Technology": "Emerging",
    "Time Period": "2020-2025",
    "Location": "Asia",
    "Functional Unit": "1 kg Aluminium Sheet",
    "Raw Material Type": "Aluminium Scrap",
    "Raw Material Quantity (kg or unit)": 100.0,
    "Energy Input Type": "Electricity",
    "Energy Input Quantity (MJ)": 250.0,
    "Processing Method": "Advanced",
    "Transport Mode": "Truck",
    "Transport Distance (km)": 300.0,
    "Fuel Type": "Diesel",
    "Metal Quality Grade": "High",
    "Material Scarcity Level": "Medium",
    "Material Cost (USD)": 500.0,
    "Processing Cost (USD)": 200.0,
    "Emissions to Air CO2 (kg)": 3081.47509765625,
    "Emissions to Air SOx (kg)": 23.762849807739258,
    "Emissions to Air NOx (kg)": 19.012903213500977,
    "Emissions to Air Particulate Matter (kg)": 11.879419326782227,
    "Emissions to Water Acid Mine Drainage (kg)": 5.273314476013184,
    "Emissions to Water Heavy Metals (kg)": 3.1632373332977295,
    "Emissions to Water BOD (kg)": 2.1091058254241943,
    "Greenhouse Gas Emissions (kg CO2-eq)": 5035.37841796875,
    "Scope 1 Emissions (kg CO2-eq)": 2504.817626953125,
    "Scope 2 Emissions (kg CO2-eq)": 1503.769775390625,
    "Scope 3 Emissions (kg CO2-eq)": 1044.5374755859375,
    "End-of-Life Treatment": "Recycling",
    "Environmental Impact Score": 56.69657897949219,
    "Metal Recyclability Factor": 0.5509214401245117,
    "Energy_per_Material": 11.394341468811035,
    "Total_Air_Emissions": 237.7609100341797,
    "Total_Water_Emissions": 10.54836368560791,
    "Transport_Intensity": 8.591459274291992,
    "GHG_per_Material": 5.105621337890625,
    "Time_Period_Numeric": 2017.448974609375,
    "Total_Cost": 780.7816162109375,
    "Circularity_Score": 44.7951545715332,
    "Circular_Economy_Index": 0.46311068534851074,
    "Recycled Content (%)": 70.25933837890625,
    "Resource Efficiency (%)": 69.86246490478516,
    "Extended Product Life (years)": 20.351980209350586,
    "Recovery Rate (%)": 87.98226165771484,
    "Reuse Potential (%)": 27.105358123779297
    }

    try:
        path = generate_report_from_dict(test_data)
        print(f" Report generated successfully: {path}")
    except Exception as e:
        print(f" Error generating report: {e}")

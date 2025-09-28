import sys
import os
import json
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER

# --- Dynamically add src/rag folder to sys.path ---
rag_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rag"))
if rag_folder not in sys.path:
    sys.path.append(rag_folder)

# --- Import the RAG pipeline run_pipeline function ---
try:
    from rag_pipline import run_pipeline  # ✅ use the function, not main()
except ImportError:
    raise ImportError(
        f"❌ Could not import 'run_pipeline' from rag_pipline.py in {rag_folder}. "
        "Make sure the file exists and contains a run_pipeline() function."
    )

class LCAReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
    
    def _create_custom_styles(self):
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=20,
            spaceAfter=30,
            textColor=colors.HexColor('#1f4e79'),
            alignment=TA_CENTER
        ))
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.HexColor('#2e5c8a')
        ))

    def generate_report(self, agent_outputs, output_filename="sustainability_report.pdf"):
        doc = SimpleDocTemplate(output_filename, pagesize=A4,
                              rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=18)
        story = []
        story.extend(self._create_title_page())
        story.append(PageBreak())

        # Dynamically add sections from RAG pipeline output
        self._add_section(story, "Executive Summary", agent_outputs.get("executive_summary"))
        self._add_section(story, "Data Quality Assessment", agent_outputs.get("data_gaps"))
        self._add_section(story, "Hotspot Analysis", agent_outputs.get("hotspots"))
        self._add_section(story, "Circularity Strategies", agent_outputs.get("strategies"))
        self._add_section(story, "Scenario Modeling", agent_outputs.get("scenarios"))
        self._add_section(story, "Compliance & Risk Assessment", agent_outputs.get("compliance"))

        # Disclaimer
        story.append(PageBreak())
        story.append(Paragraph("Appendices", self.styles['SectionHeader']))
        story.append(Paragraph(
            "This report was automatically generated using an AI-powered multi-agent RAG pipeline. "
            "Results should be validated by domain experts before being used for critical decision-making.",
            self.styles['Normal']
        ))
        doc.build(story)
        return output_filename

    def _add_section(self, story, title, content):
        story.append(Paragraph(title, self.styles['SectionHeader']))
        story.append(Paragraph(content if content else "No data available for this section.", self.styles['Normal']))
        story.append(Spacer(1, 12))

    def _create_title_page(self):
        elements = []
        title = Paragraph("Sustainability & Circularity Analysis Report", self.styles['CustomTitle'])
        elements.append(title)
        elements.append(Spacer(1, 50))

        report_data = [
            ['Report Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Prepared By:', 'AI-Powered Multi-Agent System'],
            ['Methodology:', 'Retrieval-Augmented Generation + Multi-Agent Workflow']
        ]
        report_table = Table(report_data, colWidths=[2*inch, 3*inch])
        report_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8f9fa')),
            ('TEXTCOLOR', (0,0), (-1,-1), colors.black),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 12),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        elements.append(report_table)
        return elements

def generate_report_from_json(json_file, output_file="sustainability_report.pdf"):
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"❌ JSON file not found: {json_file}")

    results = run_pipeline(json_file)  # ✅ run your multi-agent pipeline
    generator = LCAReportGenerator()
    return generator.generate_report(results, output_filename=output_file)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python report_tech.py <json_file>")
        sys.exit(1)

    json_file_path = sys.argv[1]
    report_path = generate_report_from_json(json_file_path)
    print(f"✅ Report generated successfully: {report_path}")
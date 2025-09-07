from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.widgets.markers import makeMarker
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import io
import base64

class LCAReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
    
    def _create_custom_styles(self):
        """Create custom paragraph styles for the report"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=18,
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
            textColor=colors.HexColor('#2e5c8a'),
            leftIndent=0
        ))
        
        self.styles.add(ParagraphStyle(
            name='Subsection',
            parent=self.styles['Heading3'],
            fontSize=12,
            spaceBefore=15,
            spaceAfter=8,
            textColor=colors.HexColor('#4a7c59'),
            leftIndent=20
        ))
    
    def _create_matplotlib_chart(self, chart_type, data, title, save_name):
        """Create matplotlib charts and return as image buffer"""
        plt.figure(figsize=(8, 6))
        plt.style.use('seaborn-v0_8')
        
        if chart_type == 'bar':
            plt.bar(data['labels'], data['values'], color=['#2e5c8a', '#4a7c59', '#8b5a3c'])
            plt.ylabel(data.get('ylabel', 'Values'))
        elif chart_type == 'pie':
            plt.pie(data['values'], labels=data['labels'], autopct='%1.1f%%', 
                   colors=['#2e5c8a', '#4a7c59', '#8b5a3c', '#d4af37'])
        elif chart_type == 'line':
            plt.plot(data['x'], data['y'], marker='o', linewidth=2, color='#2e5c8a')
            plt.xlabel(data.get('xlabel', 'X-axis'))
            plt.ylabel(data.get('ylabel', 'Y-axis'))
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save to buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='PNG', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        plt.close()
        
        return buffer
    
    def generate_report(self, input_data, predictions, model_performance=None, output_filename="lca_report.pdf"):
        """
        Generate comprehensive LCA report with circularity insights
        """
        doc = SimpleDocTemplate(output_filename, pagesize=A4,
                              rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=18)
        
        story = []
        
        # Title Page
        story.extend(self._create_title_page(input_data))
        story.append(PageBreak())
        
        # Executive Summary
        story.extend(self._create_executive_summary(input_data, predictions))
        story.append(PageBreak())
        
        # Input Parameters Summary
        story.extend(self._create_input_summary(input_data))
        
        # LCA Results
        story.extend(self._create_lca_results(predictions))
        
        # Circularity Analysis
        story.extend(self._create_circularity_analysis(predictions))
        
        # Environmental Impact Assessment
        story.extend(self._create_environmental_impact(input_data, predictions))
        
        # Recommendations
        story.extend(self._create_recommendations(predictions, input_data))
        
        # Model Performance (if provided)
        if model_performance:
            story.extend(self._create_model_performance(model_performance))
        
        # Appendices
        story.extend(self._create_appendices(input_data))
        
        doc.build(story)
        return output_filename
    
    def _create_title_page(self, input_data):
        """Create report title page"""
        elements = []
        
        # Main Title
        title = Paragraph("Life Cycle Assessment Report<br/>Circularity Analysis", 
                         self.styles['CustomTitle'])
        elements.append(title)
        elements.append(Spacer(1, 50))
        
        # Subtitle
        subtitle = Paragraph(f"Material: {input_data.get('Raw Material Type', 'N/A')}<br/>"
                           f"Process Stage: {input_data.get('Process Stage', 'N/A')}<br/>"
                           f"Technology: {input_data.get('Technology', 'N/A')}", 
                           self.styles['Heading2'])
        elements.append(subtitle)
        elements.append(Spacer(1, 100))
        
        # Report Info Table
        report_data = [
            ['Report Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Location:', input_data.get('Location', 'N/A')],
            ['Functional Unit:', input_data.get('Functional Unit', 'N/A')],
            ['Time Period:', input_data.get('Time Period', 'N/A')]
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
        elements.append(Spacer(1, 100))
        
        # Disclaimer
        disclaimer = Paragraph(
            "<i>This report is generated using AI/ML models for LCA estimation. "
            "Results should be validated with actual measurements where possible.</i>",
            self.styles['Normal']
        )
        elements.append(disclaimer)
        
        return elements
    
    def _create_executive_summary(self, input_data, predictions):
        """Create executive summary section"""
        elements = []
        
        elements.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        
        # Key Findings
        recycled_content = predictions.get('recycled_content', 0)
        reuse_potential = predictions.get('reuse_potential', 0)
        recovery_rate = predictions.get('recovery_rate', 0)
        
        circularity_score = (recycled_content + reuse_potential + recovery_rate) / 3
        
        summary_text = f"""
        This Life Cycle Assessment analyzes the environmental impact and circularity potential 
        of {input_data.get('Raw Material Type', 'the specified material')} processing using 
        {input_data.get('Technology', 'the specified technology')}.
        
        <b>Key Findings:</b><br/>
        • Overall Circularity Score: {circularity_score:.1f}%<br/>
        • Recycled Content: {recycled_content:.1f}%<br/>
        • Reuse Potential: {reuse_potential:.1f}%<br/>
        • Recovery Rate: {recovery_rate:.1f}%<br/>
        
        <b>Circularity Assessment:</b><br/>
        {self._get_circularity_assessment(circularity_score)}<br/>
        
        <b>Primary Recommendations:</b><br/>
        {self._get_primary_recommendations(predictions)}
        """
        
        elements.append(Paragraph(summary_text, self.styles['Normal']))
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_input_summary(self, input_data):
        """Create input parameters summary"""
        elements = []
        
        elements.append(Paragraph("Input Parameters", self.styles['SectionHeader']))
        
        # Process Parameters Table
        process_data = [
            ['Parameter', 'Value', 'Unit'],
            ['Raw Material Quantity', str(input_data.get('Raw Material Quantity (kg or unit)', 'N/A')), 'kg'],
            ['Energy Input', str(input_data.get('Energy Input Quantity (MJ)', 'N/A')), 'MJ'],
            ['Transport Distance', str(input_data.get('Transport Distance (km)', 'N/A')), 'km'],
            ['Process Stage', str(input_data.get('Process Stage', 'N/A')), '-'],
            ['Technology Type', str(input_data.get('Technology', 'N/A')), '-'],
            ['Energy Source', str(input_data.get('Energy Input Type', 'N/A')), '-']
        ]
        
        process_table = Table(process_data, colWidths=[2.5*inch, 1.5*inch, 1*inch])
        process_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2e5c8a')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 12),
            ('BACKGROUND', (0,1), (-1,-1), colors.beige),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        
        elements.append(process_table)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_lca_results(self, predictions):
        """Create LCA results section with visualizations"""
        elements = []
        
        elements.append(Paragraph("LCA Results", self.styles['SectionHeader']))
        
        # Results table
        results_data = [
            ['Indicator', 'Predicted Value', 'Performance Level'],
            ['Recycled Content (%)', f"{predictions.get('recycled_content', 0):.1f}%", 
             self._get_performance_level(predictions.get('recycled_content', 0))],
            ['Reuse Potential (%)', f"{predictions.get('reuse_potential', 0):.1f}%",
             self._get_performance_level(predictions.get('reuse_potential', 0))],
            ['Recovery Rate (%)', f"{predictions.get('recovery_rate', 0):.1f}%",
             self._get_performance_level(predictions.get('recovery_rate', 0))]
        ]
        
        results_table = Table(results_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
        results_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#4a7c59')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 12),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#f0f8f0')),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        
        elements.append(results_table)
        elements.append(Spacer(1, 30))
        
        return elements
    
    def _create_circularity_analysis(self, predictions):
        """Create circularity analysis section"""
        elements = []
        
        elements.append(Paragraph("Circularity Analysis", self.styles['SectionHeader']))
        
        # Circular Flow Opportunities
        elements.append(Paragraph("Circular Flow Opportunities", self.styles['Subsection']))
        
        recycled_content = predictions.get('recycled_content', 0)
        reuse_potential = predictions.get('reuse_potential', 0)
        recovery_rate = predictions.get('recovery_rate', 0)
        
        circularity_text = f"""
        <b>Material Flow Analysis:</b><br/>
        <br/>
        • <b>Input Circularity:</b> {recycled_content:.1f}% of material comes from recycled sources<br/>
        • <b>Use Phase Extension:</b> {reuse_potential:.1f}% potential for direct reuse<br/>
        • <b>End-of-Life Recovery:</b> {recovery_rate:.1f}% material recovery potential<br/>
        <br/>
        <b>Circular Economy Indicators:</b><br/>
        <br/>
        • <b>Material Retention Rate:</b> {((reuse_potential + recovery_rate) / 2):.1f}%<br/>
        • <b>Circularity Index:</b> {((recycled_content + reuse_potential + recovery_rate) / 3):.1f}%<br/>
        • <b>Linear vs Circular Pathway:</b> {self._compare_pathways(predictions)}<br/>
        <br/>
        <b>Resource Efficiency Opportunities:</b><br/>
        {self._get_efficiency_opportunities(predictions)}
        """
        
        elements.append(Paragraph(circularity_text, self.styles['Normal']))
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_environmental_impact(self, input_data, predictions):
        """Create environmental impact section"""
        elements = []
        
        elements.append(Paragraph("Environmental Impact Assessment", self.styles['SectionHeader']))
        
        # Impact categories
        impact_text = f"""
        <b>Climate Change Potential:</b><br/>
        Based on the circularity indicators, the carbon footprint reduction potential 
        is estimated at {self._estimate_carbon_reduction(predictions):.1f}% compared to 
        conventional linear processing.<br/>
        <br/>
        <b>Resource Depletion:</b><br/>
        With {predictions.get('recycled_content', 0):.1f}% recycled content, primary 
        resource consumption is reduced significantly.<br/>
        <br/>
        <b>Waste Generation:</b><br/>
        The {predictions.get('recovery_rate', 0):.1f}% recovery rate indicates strong 
        potential for waste minimization in the circular economy model.<br/>
        <br/>
        <b>Energy Efficiency:</b><br/>
        Recycling typically requires 60-95% less energy than primary production, 
        contributing to overall environmental benefit.
        """
        
        elements.append(Paragraph(impact_text, self.styles['Normal']))
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_recommendations(self, predictions, input_data):
        """Create actionable recommendations"""
        elements = []
        
        elements.append(Paragraph("Actionable Recommendations", self.styles['SectionHeader']))
        
        recommendations = self._generate_recommendations(predictions, input_data)
        
        elements.append(Paragraph("Priority Actions:", self.styles['Subsection']))
        for i, rec in enumerate(recommendations['priority'], 1):
            elements.append(Paragraph(f"{i}. {rec}", self.styles['Normal']))
        
        elements.append(Spacer(1, 15))
        elements.append(Paragraph("Long-term Strategies:", self.styles['Subsection']))
        for i, rec in enumerate(recommendations['longterm'], 1):
            elements.append(Paragraph(f"{i}. {rec}", self.styles['Normal']))
        
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_model_performance(self, model_performance):
        """Create model performance section"""
        elements = []
        
        elements.append(Paragraph("Model Performance Metrics", self.styles['SectionHeader']))
        
        # Performance table
        perf_data = [['Model', 'RMSE', 'MAE', 'R²']]
        for model_name, metrics in model_performance.items():
            perf_data.append([
                model_name,
                f"{metrics.get('rmse', 0):.3f}",
                f"{metrics.get('mae', 0):.3f}",
                f"{metrics.get('r2', 0):.3f}"
            ])
        
        perf_table = Table(perf_data, colWidths=[2*inch, 1*inch, 1*inch, 1*inch])
        perf_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#8b5a3c')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('BOTTOMPADDING', (0,0), (-1,-1), 12),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        
        elements.append(perf_table)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_appendices(self, input_data):
        """Create appendices section"""
        elements = []
        
        elements.append(PageBreak())
        elements.append(Paragraph("Appendices", self.styles['SectionHeader']))
        
        # Methodology
        elements.append(Paragraph("A. Methodology", self.styles['Subsection']))
        methodology_text = """
        This LCA report was generated using XGBoost machine learning models trained on 
        comprehensive LCA databases. The models predict circularity indicators based on 
        process parameters, technology choices, and material characteristics.<br/>
        <br/>
        Model Features:<br/>
        • Process stage and technology type<br/>
        • Energy input type and quantity<br/>
        • Transport distance and mode<br/>
        • Raw material characteristics<br/>
        • Geographic location factors
        """
        elements.append(Paragraph(methodology_text, self.styles['Normal']))
        
        # Data Sources
        elements.append(Paragraph("B. Data Sources and Assumptions", self.styles['Subsection']))
        datasources_text = """
        • Industry-standard LCA databases (ecoinvent, GaBi)<br/>
        • Peer-reviewed literature on metal recycling<br/>
        • Technology-specific emission factors<br/>
        • Regional energy mix considerations
        """
        elements.append(Paragraph(datasources_text, self.styles['Normal']))
        
        return elements
    
    # Helper methods
    def _get_circularity_assessment(self, score):
        if score >= 75:
            return "Excellent circular potential with strong material loops"
        elif score >= 50:
            return "Good circular potential with room for improvement"
        elif score >= 25:
            return "Moderate circular potential, requires strategic intervention"
        else:
            return "Low circular potential, significant improvements needed"
    
    def _get_performance_level(self, value):
        if value >= 75:
            return "Excellent"
        elif value >= 50:
            return "Good"
        elif value >= 25:
            return "Fair"
        else:
            return "Poor"
    
    def _get_primary_recommendations(self, predictions):
        recs = []
        if predictions.get('recycled_content', 0) < 50:
            recs.append("• Increase recycled material input")
        if predictions.get('reuse_potential', 0) < 50:
            recs.append("• Improve product design for reusability")
        if predictions.get('recovery_rate', 0) < 50:
            recs.append("• Enhance end-of-life recovery systems")
        return "<br/>".join(recs) if recs else "• Continue current best practices"
    
    def _compare_pathways(self, predictions):
        circular_score = (predictions.get('recycled_content', 0) + 
                         predictions.get('reuse_potential', 0) + 
                         predictions.get('recovery_rate', 0)) / 3
        if circular_score > 50:
            return "Circular pathway shows significant advantage"
        else:
            return "Linear pathway currently dominant, circular improvements needed"
    
    def _get_efficiency_opportunities(self, predictions):
        opportunities = []
        if predictions.get('recycled_content', 0) < 70:
            opportunities.append("• Expand recycled material sourcing networks")
        if predictions.get('reuse_potential', 0) < 60:
            opportunities.append("• Implement design for disassembly principles")
        if predictions.get('recovery_rate', 0) < 80:
            opportunities.append("• Develop advanced sorting and recovery technologies")
        
        return "<br/>".join(opportunities) if opportunities else "• Current efficiency levels are satisfactory"
    
    def _estimate_carbon_reduction(self, predictions):
        # Simplified carbon reduction estimation based on circularity
        recycling_reduction = predictions.get('recycled_content', 0) * 0.7  # 70% reduction per % recycled
        reuse_reduction = predictions.get('reuse_potential', 0) * 0.9  # 90% reduction per % reused
        return min((recycling_reduction + reuse_reduction) / 2, 85)  # Cap at 85%
    
    def _generate_recommendations(self, predictions, input_data):
        priority = []
        longterm = []
        
        # Priority recommendations based on predictions
        if predictions.get('recycled_content', 0) < 40:
            priority.append("Establish partnerships with recycled material suppliers")
        if predictions.get('reuse_potential', 0) < 30:
            priority.append("Redesign products for modularity and repairability")
        if predictions.get('recovery_rate', 0) < 50:
            priority.append("Implement take-back programs for end-of-life products")
        
        # Long-term strategies
        longterm.append("Develop closed-loop supply chain partnerships")
        longterm.append("Invest in advanced material separation technologies")
        longterm.append("Create digital material passports for traceability")
        
        return {'priority': priority, 'longterm': longterm}


# Integration function for your existing system
def generate_lca_report_from_predictions(model_predictions, input_parameters, output_file="lca_circularity_report.pdf"):
    """
    Integration function to generate report from your XGBoost model predictions
    """
   
    generator = LCAReportGenerator()
    
    # Example model performance (replace with your actual metrics)
    model_perf = {
        'Recycled Content Model': {'rmse': 7.1, 'mae': 5.5, 'r2': 0.94},
        'Reuse Potential Model': {'rmse': 7.9, 'mae': 6.3, 'r2': 0.87},
        'Recovery Rate Model': {'rmse': 3.3, 'mae': 2.6, 'r2': 0.96}
    }
    
    return generator.generate_report(
        input_data=input_parameters,
        predictions=model_predictions,
        model_performance=model_perf,
        output_filename=output_file
    )


# Example usage with your existing models
if __name__ == "__main__":
    # Example input data
    sample_input = {
        'Raw Material Type': 'Aluminium',
        'Process Stage': 'Secondary Processing',
        'Technology': 'Electric Arc Furnace',
        'Location': 'Germany',
        'Functional Unit': '1 kg Al',
        'Time Period': '2024',
        'Raw Material Quantity (kg or unit)': 1000,
        'Energy Input Quantity (MJ)': 15000,
        'Transport Distance (km)': 500,
        'Energy Input Type': 'Grid Electricity'
    }
    
    # Example predictions
    sample_predictions = {
        'recycled_content': 65.2,
        'reuse_potential': 45.8,
        'recovery_rate': 78.3
    }
    
    # Generate report
    report_path = generate_lca_report_from_predictions(
        sample_predictions, 
        sample_input, 
        "sample_lca_report.pdf"
    )
    
    print(f"Report generated: {report_path}")
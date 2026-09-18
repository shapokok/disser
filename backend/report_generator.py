"""
PDF Report Generation for Crop Disease Analysis
Creates professional reports with images, predictions, and visualizations
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime
import io
import base64
from PIL import Image


def create_pdf_report(analysis_data, output_path=None):
    """
    Create a comprehensive PDF report for crop disease analysis

    Args:
        analysis_data: Dictionary containing analysis results
        output_path: Path to save PDF (if None, returns bytes)

    Returns:
        PDF bytes if output_path is None, else None (saves to file)
    """
    # Create PDF buffer
    if output_path:
        doc = SimpleDocTemplate(output_path, pagesize=letter)
    else:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)

    # Container for PDF elements
    story = []

    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2ecc71'),
        spaceAfter=30,
        alignment=TA_CENTER
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=12
    )

    # Title
    story.append(Paragraph("Crop Disease Detection Report", title_style))
    story.append(Spacer(1, 0.2*inch))

    # Report metadata
    metadata_style = ParagraphStyle(
        'metadata',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.grey,
        alignment=TA_CENTER
    )

    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        metadata_style
    ))
    story.append(Spacer(1, 0.3*inch))

    # Divider line
    story.append(Table(
        [['']], colWidths=[6.5*inch],
        style=[('LINEABOVE', (0,0), (-1,0), 2, colors.HexColor('#2ecc71'))]
    ))
    story.append(Spacer(1, 0.3*inch))

    # Analysis Summary
    story.append(Paragraph("Analysis Summary", heading_style))

    summary_data = [
        ['Property', 'Value'],
        ['Image Name', analysis_data.get('image_name', 'N/A')],
        ['Model Used', analysis_data.get('model_used', 'N/A')],
        ['Explanation Method', analysis_data.get('explanation_method', 'N/A')],
        ['Dataset Type', analysis_data.get('dataset_type', 'N/A')],
        ['Inference Time', f"{analysis_data.get('inference_time_ms', 'N/A')} ms"],
    ]

    summary_table = Table(summary_data, colWidths=[2*inch, 4*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ecc71')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
    ]))

    story.append(summary_table)
    story.append(Spacer(1, 0.3*inch))

    # Prediction Results
    story.append(Paragraph("Prediction Results", heading_style))

    prediction = analysis_data.get('prediction', {})
    confidence = prediction.get('confidence', 0)

    # Confidence color
    if confidence > 0.8:
        conf_color = colors.HexColor('#2ecc71')
    elif confidence > 0.5:
        conf_color = colors.HexColor('#f39c12')
    else:
        conf_color = colors.HexColor('#e74c3c')

    prediction_data = [
        ['Metric', 'Value'],
        ['Predicted Disease', prediction.get('class', 'Unknown')],
        ['Confidence Score', prediction.get('confidence_percent', '0%')],
    ]

    prediction_table = Table(prediction_data, colWidths=[2*inch, 4*inch])
    prediction_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (0, -1), colors.lightgrey),
        ('BACKGROUND', (1, 1), (1, 1), colors.lightblue),
        ('BACKGROUND', (1, 2), (1, 2), conf_color),
        ('TEXTCOLOR', (1, 2), (1, 2), colors.whitesmoke),
        ('FONTNAME', (1, 1), (1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (1, 1), (1, -1), 14),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))

    story.append(prediction_table)
    story.append(Spacer(1, 0.3*inch))

    # Top 3 Predictions
    top_predictions = analysis_data.get('top_predictions', [])
    if top_predictions:
        story.append(Paragraph("Top 3 Predictions", heading_style))

        top_pred_data = [['Rank', 'Disease Class', 'Confidence']]
        for i, pred in enumerate(top_predictions[:3], 1):
            top_pred_data.append([
                str(i),
                pred.get('class', 'Unknown'),
                pred.get('confidence_percent', '0%')
            ])

        top_pred_table = Table(top_pred_data, colWidths=[0.8*inch, 3.7*inch, 1.5*inch])
        top_pred_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ]))

        story.append(top_pred_table)
        story.append(Spacer(1, 0.3*inch))

    # Visualization (if available)
    visualization_base64 = analysis_data.get('visualization')
    if visualization_base64:
        story.append(PageBreak())
        story.append(Paragraph("Explainable AI Visualization", heading_style))
        story.append(Spacer(1, 0.2*inch))

        try:
            # Decode base64 image
            img_data = base64.b64decode(visualization_base64)
            img = Image.open(io.BytesIO(img_data))

            # Save to temp buffer
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            img_buffer.seek(0)

            # Add to PDF
            rl_img = RLImage(img_buffer, width=6*inch, height=None)
            story.append(rl_img)
            story.append(Spacer(1, 0.2*inch))

            # Explanation text
            explanation_text = f"""
            The visualization above shows the Grad-CAM (Gradient-weighted Class Activation Mapping)
            or LIME (Local Interpretable Model-agnostic Explanations) heatmap. The colored regions
            indicate which parts of the image the AI model focused on when making its prediction.
            Warmer colors (red/yellow) indicate higher importance in the decision-making process.
            """
            story.append(Paragraph(explanation_text, styles['Normal']))

        except Exception as e:
            story.append(Paragraph(f"Error loading visualization: {str(e)}", styles['Normal']))

    # Recommendations section
    story.append(PageBreak())
    story.append(Paragraph("Recommendations & Next Steps", heading_style))

    recommendations = get_recommendations(prediction.get('class_raw', ''), confidence)
    for rec in recommendations:
        story.append(Paragraph(f"• {rec}", styles['Normal']))
        story.append(Spacer(1, 0.1*inch))

    # Footer
    story.append(Spacer(1, 0.5*inch))
    story.append(Table(
        [['']], colWidths=[6.5*inch],
        style=[('LINEABOVE', (0,0), (-1,0), 1, colors.grey)]
    ))
    footer_style = ParagraphStyle(
        'footer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.grey,
        alignment=TA_CENTER
    )
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(
        "Crop Disease Detection System | Master's Thesis Project | For Educational Purposes",
        footer_style
    ))
    story.append(Paragraph(
        f"© 2025 - Generated on {datetime.now().strftime('%Y-%m-%d')}",
        footer_style
    ))

    # Build PDF
    doc.build(story)

    if output_path:
        return None
    else:
        pdf_bytes = buffer.getvalue()
        buffer.close()
        return pdf_bytes


def get_recommendations(disease_class, confidence):
    """
    Get recommendations based on detected disease

    Args:
        disease_class: Detected disease class
        confidence: Confidence score

    Returns:
        List of recommendation strings
    """
    recommendations = []

    if confidence < 0.5:
        recommendations.append(
            "⚠️ Low confidence detection. Consider retaking the image with better lighting and focus."
        )
        recommendations.append(
            "Ensure the leaf is clearly visible without blur or obstructions."
        )

    if 'healthy' in disease_class.lower():
        recommendations.append(
            "✓ Plant appears healthy. Continue regular monitoring and maintenance."
        )
        recommendations.append(
            "Maintain current watering and fertilization practices."
        )
    else:
        recommendations.append(
            "⚠️ Disease detected. Isolate affected plants to prevent spread."
        )
        recommendations.append(
            "Consult with an agricultural expert for treatment recommendations."
        )
        recommendations.append(
            "Consider removing severely infected leaves."
        )

        if 'blight' in disease_class.lower():
            recommendations.append(
                "For blight: Improve air circulation and reduce moisture on leaves."
            )
            recommendations.append(
                "Apply appropriate fungicide as recommended by local agricultural extension."
            )
        elif 'rust' in disease_class.lower():
            recommendations.append(
                "For rust: Remove infected plant parts and apply fungicides."
            )
        elif 'spot' in disease_class.lower():
            recommendations.append(
                "For spot diseases: Improve air circulation and avoid overhead watering."
            )

    recommendations.append(
        "📸 Take multiple photos from different angles for comprehensive assessment."
    )
    recommendations.append(
        "📊 Monitor the affected area regularly and track disease progression."
    )

    return recommendations


def generate_comparison_report(comparison_data, output_path=None):
    """
    Generate PDF report for model comparison

    Args:
        comparison_data: Dictionary with comparison results
        output_path: Path to save PDF

    Returns:
        PDF bytes or None
    """
    if output_path:
        doc = SimpleDocTemplate(output_path, pagesize=letter)
    else:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)

    story = []
    styles = getSampleStyleSheet()

    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=22,
        textColor=colors.HexColor('#2ecc71'),
        spaceAfter=20,
        alignment=TA_CENTER
    )

    story.append(Paragraph("Model Comparison Report", title_style))
    story.append(Spacer(1, 0.3*inch))

    # Comparison table
    comparisons = comparison_data.get('comparisons', {})

    comp_data = [['Model', 'Prediction', 'Confidence', 'Inference Time']]
    for model, result in comparisons.items():
        comp_data.append([
            model.upper(),
            result.get('predicted_class', 'N/A'),
            result.get('confidence_percent', '0%'),
            f"{result.get('inference_time_ms', 0):.2f} ms"
        ])

    comp_table = Table(comp_data, colWidths=[1.5*inch, 2.5*inch, 1.3*inch, 1.2*inch])
    comp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ecc71')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))

    story.append(comp_table)

    # Build PDF
    doc.build(story)

    if output_path:
        return None
    else:
        pdf_bytes = buffer.getvalue()
        buffer.close()
        return pdf_bytes

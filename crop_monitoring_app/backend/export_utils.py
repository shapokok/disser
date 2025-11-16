"""
Excel Export Utilities for Crop Disease Detection System
Provides functions to export analysis results, batch predictions, and model comparisons to Excel
"""

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.chart import BarChart, Reference, PieChart
from datetime import datetime
import io


def export_to_excel(data, export_type='analysis', output_path=None):
    """
    Export data to Excel format

    Args:
        data: Dictionary containing data to export
        export_type: Type of export ('analysis', 'batch', 'comparison', 'validation')
        output_path: Path to save Excel file (if None, returns bytes)

    Returns:
        Excel bytes if output_path is None, else None (saves to file)
    """
    wb = Workbook()

    if export_type == 'analysis':
        _create_analysis_sheet(wb, data)
    elif export_type == 'batch':
        _create_batch_sheet(wb, data)
    elif export_type == 'comparison':
        _create_comparison_sheet(wb, data)
    elif export_type == 'validation':
        _create_validation_sheet(wb, data)
    else:
        raise ValueError(f"Unknown export type: {export_type}")

    # Save or return bytes
    if output_path:
        wb.save(output_path)
        return None
    else:
        buffer = io.BytesIO()
        wb.save(buffer)
        excel_bytes = buffer.getvalue()
        buffer.close()
        return excel_bytes


def _create_analysis_sheet(wb, data):
    """Create worksheet for single image analysis results"""
    ws = wb.active
    ws.title = "Analysis Report"

    # Styles
    header_fill = PatternFill(start_color="2ECC71", end_color="2ECC71", fill_type="solid")
    header_font = Font(bold=True, color="FFFFFF", size=12)
    title_font = Font(bold=True, size=16, color="2C3E50")
    section_font = Font(bold=True, size=13, color="34495E")
    border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )

    # Title
    ws.merge_cells('A1:D1')
    ws['A1'] = "Crop Disease Detection - Analysis Report"
    ws['A1'].font = title_font
    ws['A1'].alignment = Alignment(horizontal='center', vertical='center')
    ws.row_dimensions[1].height = 25

    # Metadata
    ws['A2'] = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ws['A2'].font = Font(italic=True, color="7F8C8D", size=10)
    ws.merge_cells('A2:D2')

    # Analysis Summary Section
    row = 4
    ws[f'A{row}'] = "Analysis Summary"
    ws[f'A{row}'].font = section_font
    ws.merge_cells(f'A{row}:B{row}')
    row += 1

    # Summary data
    summary_items = [
        ('Image Name', data.get('image_name', 'N/A')),
        ('Model Used', data.get('model_used', 'N/A')),
        ('Explanation Method', data.get('explanation_method', 'N/A')),
        ('Dataset Type', data.get('dataset_type', 'N/A')),
        ('Inference Time', f"{data.get('inference_time_ms', 'N/A')} ms"),
    ]

    for label, value in summary_items:
        ws[f'A{row}'] = label
        ws[f'B{row}'] = value
        ws[f'A{row}'].font = Font(bold=True)
        ws[f'A{row}'].fill = PatternFill(start_color="ECF0F1", end_color="ECF0F1", fill_type="solid")
        ws[f'A{row}'].border = border
        ws[f'B{row}'].border = border
        row += 1

    # Prediction Results Section
    row += 1
    ws[f'A{row}'] = "Prediction Results"
    ws[f'A{row}'].font = section_font
    ws.merge_cells(f'A{row}:B{row}')
    row += 1

    prediction = data.get('prediction', {})
    confidence = prediction.get('confidence', 0)

    # Confidence color coding
    if confidence > 0.8:
        conf_color = "2ECC71"  # Green
    elif confidence > 0.5:
        conf_color = "F39C12"  # Orange
    else:
        conf_color = "E74C3C"  # Red

    ws[f'A{row}'] = "Predicted Disease"
    ws[f'B{row}'] = prediction.get('class', 'Unknown')
    ws[f'A{row}'].font = Font(bold=True)
    ws[f'A{row}'].fill = PatternFill(start_color="ECF0F1", end_color="ECF0F1", fill_type="solid")
    ws[f'B{row}'].font = Font(bold=True, size=12)
    ws[f'B{row}'].fill = PatternFill(start_color="D6EAF8", end_color="D6EAF8", fill_type="solid")
    ws[f'A{row}'].border = border
    ws[f'B{row}'].border = border
    row += 1

    ws[f'A{row}'] = "Confidence Score"
    ws[f'B{row}'] = prediction.get('confidence_percent', '0%')
    ws[f'A{row}'].font = Font(bold=True)
    ws[f'A{row}'].fill = PatternFill(start_color="ECF0F1", end_color="ECF0F1", fill_type="solid")
    ws[f'B{row}'].font = Font(bold=True, size=12, color="FFFFFF")
    ws[f'B{row}'].fill = PatternFill(start_color=conf_color, end_color=conf_color, fill_type="solid")
    ws[f'A{row}'].border = border
    ws[f'B{row}'].border = border
    row += 1

    # Top 3 Predictions Section
    top_predictions = data.get('top_predictions', [])
    if top_predictions:
        row += 1
        ws[f'A{row}'] = "Top 3 Predictions"
        ws[f'A{row}'].font = section_font
        ws.merge_cells(f'A{row}:C{row}')
        row += 1

        # Headers
        ws[f'A{row}'] = "Rank"
        ws[f'B{row}'] = "Disease Class"
        ws[f'C{row}'] = "Confidence"
        for col in ['A', 'B', 'C']:
            ws[f'{col}{row}'].font = header_font
            ws[f'{col}{row}'].fill = header_fill
            ws[f'{col}{row}'].alignment = Alignment(horizontal='center')
            ws[f'{col}{row}'].border = border
        row += 1

        # Data
        for i, pred in enumerate(top_predictions[:3], 1):
            ws[f'A{row}'] = i
            ws[f'B{row}'] = pred.get('class', 'Unknown')
            ws[f'C{row}'] = pred.get('confidence_percent', '0%')
            ws[f'A{row}'].alignment = Alignment(horizontal='center')
            ws[f'C{row}'].alignment = Alignment(horizontal='center')
            for col in ['A', 'B', 'C']:
                ws[f'{col}{row}'].border = border
            row += 1

    # Adjust column widths
    ws.column_dimensions['A'].width = 25
    ws.column_dimensions['B'].width = 40
    ws.column_dimensions['C'].width = 20


def _create_batch_sheet(wb, data):
    """Create worksheet for batch processing results"""
    ws = wb.active
    ws.title = "Batch Results"

    # Styles
    header_fill = PatternFill(start_color="3498DB", end_color="3498DB", fill_type="solid")
    header_font = Font(bold=True, color="FFFFFF", size=11)
    title_font = Font(bold=True, size=16, color="2C3E50")
    border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )

    # Title
    ws.merge_cells('A1:E1')
    ws['A1'] = "Batch Processing Results"
    ws['A1'].font = title_font
    ws['A1'].alignment = Alignment(horizontal='center', vertical='center')
    ws.row_dimensions[1].height = 25

    # Metadata
    ws['A2'] = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ws['A2'].font = Font(italic=True, color="7F8C8D", size=10)
    ws.merge_cells('A2:E2')

    ws['A3'] = f"Total Images: {data.get('total', 0)}"
    ws['A3'].font = Font(bold=True)
    ws.merge_cells('A3:E3')

    # Headers
    row = 5
    headers = ['#', 'Image Path', 'Predicted Class', 'Confidence', 'Status']
    for col_idx, header in enumerate(headers, 1):
        cell = ws.cell(row=row, column=col_idx, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal='center')
        cell.border = border

    # Data
    results = data.get('results', [])
    for idx, result in enumerate(results, 1):
        row += 1

        # Index
        ws.cell(row=row, column=1, value=idx).alignment = Alignment(horizontal='center')

        # Image path
        ws.cell(row=row, column=2, value=result.get('image_path', 'N/A'))

        # Check if success or error
        if 'error' in result:
            ws.cell(row=row, column=3, value='ERROR')
            ws.cell(row=row, column=4, value='-')
            ws.cell(row=row, column=5, value=result.get('error', 'Unknown error'))
            # Highlight error rows
            for col in range(1, 6):
                ws.cell(row=row, column=col).fill = PatternFill(
                    start_color="FADBD8", end_color="FADBD8", fill_type="solid"
                )
        else:
            ws.cell(row=row, column=3, value=result.get('predicted_class', 'Unknown'))
            ws.cell(row=row, column=4, value=result.get('confidence', 0))
            ws.cell(row=row, column=5, value='Success')
            # Highlight success rows
            for col in range(1, 6):
                ws.cell(row=row, column=col).fill = PatternFill(
                    start_color="D5F4E6", end_color="D5F4E6", fill_type="solid"
                )

        # Apply borders
        for col in range(1, 6):
            ws.cell(row=row, column=col).border = border

    # Adjust column widths
    ws.column_dimensions['A'].width = 8
    ws.column_dimensions['B'].width = 50
    ws.column_dimensions['C'].width = 35
    ws.column_dimensions['D'].width = 15
    ws.column_dimensions['E'].width = 15


def _create_comparison_sheet(wb, data):
    """Create worksheet for model comparison results"""
    ws = wb.active
    ws.title = "Model Comparison"

    # Styles
    header_fill = PatternFill(start_color="2ECC71", end_color="2ECC71", fill_type="solid")
    header_font = Font(bold=True, color="FFFFFF", size=11)
    title_font = Font(bold=True, size=16, color="2C3E50")
    border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )

    # Title
    ws.merge_cells('A1:D1')
    ws['A1'] = "Model Comparison Report"
    ws['A1'].font = title_font
    ws['A1'].alignment = Alignment(horizontal='center', vertical='center')
    ws.row_dimensions[1].height = 25

    # Metadata
    ws['A2'] = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ws['A2'].font = Font(italic=True, color="7F8C8D", size=10)
    ws.merge_cells('A2:D2')

    ws['A3'] = f"Image: {data.get('image_path', 'N/A')}"
    ws['A3'].font = Font(bold=True)
    ws.merge_cells('A3:D3')

    # Headers
    row = 5
    headers = ['Model', 'Prediction', 'Confidence', 'Inference Time (ms)']
    for col_idx, header in enumerate(headers, 1):
        cell = ws.cell(row=row, column=col_idx, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal='center')
        cell.border = border

    # Data
    comparisons = data.get('comparisons', {})
    for model_name, result in comparisons.items():
        row += 1
        ws.cell(row=row, column=1, value=model_name.upper())
        ws.cell(row=row, column=2, value=result.get('predicted_class', 'N/A'))
        ws.cell(row=row, column=3, value=result.get('confidence_percent', '0%'))
        ws.cell(row=row, column=4, value=result.get('inference_time_ms', 0))

        # Center alignment
        ws.cell(row=row, column=3).alignment = Alignment(horizontal='center')
        ws.cell(row=row, column=4).alignment = Alignment(horizontal='center')

        # Apply borders
        for col in range(1, 5):
            ws.cell(row=row, column=col).border = border

    # Adjust column widths
    ws.column_dimensions['A'].width = 20
    ws.column_dimensions['B'].width = 40
    ws.column_dimensions['C'].width = 18
    ws.column_dimensions['D'].width = 22


def _create_validation_sheet(wb, data):
    """Create worksheet for model validation metrics"""
    ws = wb.active
    ws.title = "Validation Report"

    # Styles
    header_fill = PatternFill(start_color="9B59B6", end_color="9B59B6", fill_type="solid")
    header_font = Font(bold=True, color="FFFFFF", size=11)
    title_font = Font(bold=True, size=16, color="2C3E50")
    section_font = Font(bold=True, size=13, color="34495E")
    border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )

    # Title
    ws.merge_cells('A1:D1')
    ws['A1'] = "Model Validation Report"
    ws['A1'].font = title_font
    ws['A1'].alignment = Alignment(horizontal='center', vertical='center')
    ws.row_dimensions[1].height = 25

    # Metadata
    ws['A2'] = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ws['A2'].font = Font(italic=True, color="7F8C8D", size=10)
    ws.merge_cells('A2:D2')

    # Overall Metrics Section
    row = 4
    ws[f'A{row}'] = "Overall Performance Metrics"
    ws[f'A{row}'].font = section_font
    ws.merge_cells(f'A{row}:B{row}')
    row += 1

    # Overall metrics
    overall_metrics = data.get('overall_metrics', {})
    metrics_items = [
        ('Accuracy', f"{overall_metrics.get('accuracy', 0):.4f}"),
        ('Precision', f"{overall_metrics.get('precision', 0):.4f}"),
        ('Recall', f"{overall_metrics.get('recall', 0):.4f}"),
        ('F1-Score', f"{overall_metrics.get('f1_score', 0):.4f}"),
    ]

    for label, value in metrics_items:
        ws[f'A{row}'] = label
        ws[f'B{row}'] = value
        ws[f'A{row}'].font = Font(bold=True)
        ws[f'A{row}'].fill = PatternFill(start_color="ECF0F1", end_color="ECF0F1", fill_type="solid")
        ws[f'A{row}'].border = border
        ws[f'B{row}'].border = border
        ws[f'B{row}'].alignment = Alignment(horizontal='center')
        row += 1

    # Per-class metrics section
    per_class = data.get('per_class_metrics', [])
    if per_class:
        row += 1
        ws[f'A{row}'] = "Per-Class Metrics"
        ws[f'A{row}'].font = section_font
        ws.merge_cells(f'A{row}:E{row}')
        row += 1

        # Headers
        headers = ['Class', 'Precision', 'Recall', 'F1-Score', 'Support']
        for col_idx, header in enumerate(headers, 1):
            cell = ws.cell(row=row, column=col_idx, value=header)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal='center')
            cell.border = border
        row += 1

        # Data
        for class_metric in per_class:
            ws.cell(row=row, column=1, value=class_metric.get('class_name', 'Unknown'))
            ws.cell(row=row, column=2, value=f"{class_metric.get('precision', 0):.4f}")
            ws.cell(row=row, column=3, value=f"{class_metric.get('recall', 0):.4f}")
            ws.cell(row=row, column=4, value=f"{class_metric.get('f1_score', 0):.4f}")
            ws.cell(row=row, column=5, value=class_metric.get('support', 0))

            # Center alignment for metrics
            for col in range(2, 6):
                ws.cell(row=row, column=col).alignment = Alignment(horizontal='center')
                ws.cell(row=row, column=col).border = border
            ws.cell(row=row, column=1).border = border
            row += 1

    # Adjust column widths
    ws.column_dimensions['A'].width = 40
    ws.column_dimensions['B'].width = 15
    ws.column_dimensions['C'].width = 15
    ws.column_dimensions['D'].width = 15
    ws.column_dimensions['E'].width = 15


def create_analysis_excel(analysis_data, output_path=None):
    """
    Convenience function to create Excel file for analysis results

    Args:
        analysis_data: Dictionary with analysis results
        output_path: Optional file path to save

    Returns:
        Excel bytes or None
    """
    return export_to_excel(analysis_data, export_type='analysis', output_path=output_path)


def create_batch_excel(batch_data, output_path=None):
    """
    Convenience function to create Excel file for batch results

    Args:
        batch_data: Dictionary with batch processing results
        output_path: Optional file path to save

    Returns:
        Excel bytes or None
    """
    return export_to_excel(batch_data, export_type='batch', output_path=output_path)


def create_comparison_excel(comparison_data, output_path=None):
    """
    Convenience function to create Excel file for model comparison

    Args:
        comparison_data: Dictionary with model comparison results
        output_path: Optional file path to save

    Returns:
        Excel bytes or None
    """
    return export_to_excel(comparison_data, export_type='comparison', output_path=output_path)


def create_validation_excel(validation_data, output_path=None):
    """
    Convenience function to create Excel file for validation report

    Args:
        validation_data: Dictionary with validation metrics
        output_path: Optional file path to save

    Returns:
        Excel bytes or None
    """
    return export_to_excel(validation_data, export_type='validation', output_path=output_path)

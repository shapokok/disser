"""
Export utilities for generating CSV, JSON, and Excel reports
from crop disease detection results
"""

import csv
import json
import io
from datetime import datetime
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter


def export_to_csv(results):
    """
    Export analysis results to CSV format

    Args:
        results: List of analysis results or single result dict

    Returns:
        CSV string data
    """
    # Ensure results is a list
    if isinstance(results, dict):
        results = [results]

    # Create CSV in memory
    output = io.StringIO()

    # Define CSV fields
    fieldnames = [
        'timestamp',
        'image_name',
        'predicted_class',
        'confidence',
        'model_used',
        'explanation_method',
        'inference_time_ms',
        'top1_class',
        'top1_confidence',
        'top2_class',
        'top2_confidence',
        'top3_class',
        'top3_confidence'
    ]

    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    # Write each result
    for result in results:
        # Extract top predictions
        top_predictions = result.get('top_predictions', [])

        row = {
            'timestamp': result.get('timestamp', datetime.now().isoformat()),
            'image_name': result.get('image_name', result.get('original_name', 'N/A')),
            'predicted_class': result.get('prediction', {}).get('class', 'N/A'),
            'confidence': result.get('prediction', {}).get('confidence', 0),
            'model_used': result.get('model_used', 'N/A'),
            'explanation_method': result.get('explanation_method', 'N/A'),
            'inference_time_ms': result.get('inference_time_ms', 0),
        }

        # Add top-3 predictions
        for i in range(3):
            if i < len(top_predictions):
                row[f'top{i+1}_class'] = top_predictions[i].get('class', 'N/A')
                row[f'top{i+1}_confidence'] = top_predictions[i].get('confidence', 0)
            else:
                row[f'top{i+1}_class'] = ''
                row[f'top{i+1}_confidence'] = ''

        writer.writerow(row)

    return output.getvalue()


def export_to_json(results, pretty=True):
    """
    Export analysis results to JSON format

    Args:
        results: List of analysis results or single result dict
        pretty: Whether to format JSON with indentation

    Returns:
        JSON string data
    """
    # Ensure results is a list
    if isinstance(results, dict):
        results = [results]

    # Create export structure
    export_data = {
        'export_timestamp': datetime.now().isoformat(),
        'total_results': len(results),
        'results': []
    }

    # Process each result
    for result in results:
        export_result = {
            'timestamp': result.get('timestamp', datetime.now().isoformat()),
            'image_name': result.get('image_name', result.get('original_name', 'N/A')),
            'prediction': {
                'class': result.get('prediction', {}).get('class', 'N/A'),
                'class_raw': result.get('prediction', {}).get('class_raw', 'N/A'),
                'confidence': result.get('prediction', {}).get('confidence', 0),
                'confidence_percent': result.get('prediction', {}).get('confidence_percent', '0%')
            },
            'model_used': result.get('model_used', 'N/A'),
            'explanation_method': result.get('explanation_method', 'N/A'),
            'inference_time_ms': result.get('inference_time_ms', 0),
            'top_predictions': result.get('top_predictions', [])
        }

        export_data['results'].append(export_result)

    # Convert to JSON
    if pretty:
        return json.dumps(export_data, indent=2)
    else:
        return json.dumps(export_data)


def export_to_excel(results):
    """
    Export analysis results to Excel format with formatting

    Args:
        results: List of analysis results or single result dict

    Returns:
        Excel file bytes
    """
    # Ensure results is a list
    if isinstance(results, dict):
        results = [results]

    # Create workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Analysis Results"

    # Define styles
    header_font = Font(bold=True, color="FFFFFF", size=12)
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    header_alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )

    # Headers
    headers = [
        'Timestamp',
        'Image Name',
        'Predicted Class',
        'Confidence',
        'Model Used',
        'Explanation',
        'Inference Time (ms)',
        'Top 1 Class',
        'Top 1 Conf.',
        'Top 2 Class',
        'Top 2 Conf.',
        'Top 3 Class',
        'Top 3 Conf.'
    ]

    # Write headers
    for col_num, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col_num)
        cell.value = header
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_alignment
        cell.border = border

    # Write data
    for row_num, result in enumerate(results, 2):
        # Get top predictions
        top_predictions = result.get('top_predictions', [])

        # Row data
        row_data = [
            result.get('timestamp', datetime.now().isoformat()),
            result.get('image_name', result.get('original_name', 'N/A')),
            result.get('prediction', {}).get('class', 'N/A'),
            result.get('prediction', {}).get('confidence', 0),
            result.get('model_used', 'N/A'),
            result.get('explanation_method', 'N/A'),
            result.get('inference_time_ms', 0),
        ]

        # Add top-3 predictions
        for i in range(3):
            if i < len(top_predictions):
                row_data.append(top_predictions[i].get('class', 'N/A'))
                row_data.append(top_predictions[i].get('confidence', 0))
            else:
                row_data.append('')
                row_data.append('')

        # Write row
        for col_num, value in enumerate(row_data, 1):
            cell = ws.cell(row=row_num, column=col_num)
            cell.value = value
            cell.border = border
            cell.alignment = Alignment(vertical="center")

            # Format confidence as percentage
            if col_num in [4, 9, 11, 13]:  # Confidence columns
                if isinstance(value, (int, float)) and value > 0:
                    cell.number_format = '0.00%'

    # Auto-adjust column widths
    for col_num in range(1, len(headers) + 1):
        column_letter = get_column_letter(col_num)
        max_length = 0

        for cell in ws[column_letter]:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass

        adjusted_width = min(max_length + 2, 50)
        ws.column_dimensions[column_letter].width = adjusted_width

    # Freeze header row
    ws.freeze_panes = 'A2'

    # Save to bytes
    output = io.BytesIO()
    wb.save(output)
    output.seek(0)

    return output.getvalue()


def export_comparison_to_csv(comparison_data):
    """
    Export model comparison results to CSV

    Args:
        comparison_data: Model comparison results dict

    Returns:
        CSV string data
    """
    output = io.StringIO()

    fieldnames = [
        'model_name',
        'predicted_class',
        'confidence',
        'confidence_percent',
        'inference_time_ms'
    ]

    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    # Get comparisons
    comparisons = comparison_data.get('comparisons', {})

    for model_name, result in comparisons.items():
        row = {
            'model_name': model_name,
            'predicted_class': result.get('predicted_class', 'N/A'),
            'confidence': result.get('confidence', 0),
            'confidence_percent': result.get('confidence_percent', '0%'),
            'inference_time_ms': result.get('inference_time_ms', 0)
        }
        writer.writerow(row)

    return output.getvalue()


def export_comparison_to_excel(comparison_data):
    """
    Export model comparison results to Excel with formatting

    Args:
        comparison_data: Model comparison results dict

    Returns:
        Excel file bytes
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "Model Comparison"

    # Styles
    header_font = Font(bold=True, color="FFFFFF", size=12)
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    header_alignment = Alignment(horizontal="center", vertical="center")

    border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )

    # Headers
    headers = ['Model', 'Predicted Class', 'Confidence', 'Confidence %', 'Inference Time (ms)']

    for col_num, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col_num)
        cell.value = header
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_alignment
        cell.border = border

    # Data
    comparisons = comparison_data.get('comparisons', {})

    for row_num, (model_name, result) in enumerate(comparisons.items(), 2):
        row_data = [
            model_name.upper(),
            result.get('predicted_class', 'N/A'),
            result.get('confidence', 0),
            result.get('confidence_percent', '0%'),
            result.get('inference_time_ms', 0)
        ]

        for col_num, value in enumerate(row_data, 1):
            cell = ws.cell(row=row_num, column=col_num)
            cell.value = value
            cell.border = border
            cell.alignment = Alignment(vertical="center")

            # Format confidence column as percentage
            if col_num == 3 and isinstance(value, (int, float)):
                cell.number_format = '0.00%'

    # Auto-adjust columns
    for col_num in range(1, len(headers) + 1):
        column_letter = get_column_letter(col_num)
        ws.column_dimensions[column_letter].width = 20

    # Freeze header
    ws.freeze_panes = 'A2'

    # Save to bytes
    output = io.BytesIO()
    wb.save(output)
    output.seek(0)

    return output.getvalue()

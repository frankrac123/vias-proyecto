"""
Professional traffic data export module
Generates Excel and CSV reports with engineering-grade traffic analysis
"""
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import json

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False


class TrafficExporter:
    """Export traffic data to Excel and CSV formats"""
    
    def __init__(self, export_dir: str = "exports"):
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
    
    def export_session(self, session_data: Dict, lines_manager, format: str = "both") -> Dict[str, Path]:
        """
        Export session data to Excel and/or CSV
        
        Args:
            session_data: Session information (id, name, location, start_time, etc)
            lines_manager: LinesManager instance with all data
            format: 'excel', 'csv', or 'both'
        
        Returns:
            Dict mapping format -> file path
        """
        results = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format in ["excel", "both"] and HAS_OPENPYXL:
            excel_path = self._export_excel(session_data, lines_manager, timestamp)
            results['excel'] = excel_path
        
        if format in ["csv", "both"]:
            csv_path = self._export_csv(session_data, lines_manager, timestamp)
            results['csv'] = csv_path
        
        return results
    
    def _export_excel(self, session_data: Dict, lines_manager, timestamp: str) -> Path:
        """Export to professional Excel workbook"""
        wb = Workbook()
        wb.remove(wb.active)  # Remove default sheet
        
        # Define styles
        header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        header_font = Font(bold=True, color="FFFFFF", size=11)
        title_font = Font(bold=True, size=14)
        subheader_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
        subheader_font = Font(bold=True, size=10)
        border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )
        
        # Sheet 1: Summary
        ws_summary = wb.create_sheet("RESUMEN", 0)
        row = 1
        
        ws_summary[f'A{row}'] = "TRAFFIC VISION - REPORTE DE AFORO VEHICULAR"
        ws_summary[f'A{row}'].font = title_font
        row += 2
        
        summary_data = [
            ("Estudio:", session_data.get('name', 'N/A')),
            ("Ubicación:", session_data.get('location', 'N/A')),
            ("Observador:", session_data.get('observer', 'N/A')),
            ("Fecha/Hora Inicio:", session_data.get('start_time', 'N/A')),
            ("Duración:", session_data.get('duration', 'N/A')),
            ("Descripción:", session_data.get('description', '')),
        ]
        
        for label, value in summary_data:
            ws_summary[f'A{row}'] = label
            ws_summary[f'A{row}'].font = subheader_font
            ws_summary[f'B{row}'] = value
            row += 1
        
        row += 1
        
        # Summary by line
        for line_id, line in lines_manager.get_lines().items():
            ws_summary[f'A{row}'] = f"LÍNEA: {line.name}"
            ws_summary[f'A{row}'].font = subheader_font
            row += 1
            
            ws_summary[f'A{row}'] = "Intervalo"
            ws_summary[f'B{row}'] = "Volumen"
            ws_summary[f'C{row}'] = "Flujo (veh/h)"
            ws_summary[f'D{row}'] = f"{line.direction_a}→{line.direction_b}"
            ws_summary[f'E{row}'] = f"{line.direction_b}→{line.direction_a}"
            
            for cell in [f'A{row}', f'B{row}', f'C{row}', f'D{row}', f'E{row}']:
                ws_summary[cell].fill = header_fill
                ws_summary[cell].font = header_font
            row += 1
            
            for interval in [1, 5, 15, 60]:
                stats = lines_manager.get_stats(line_id, interval)
                if stats:
                    ws_summary[f'A{row}'] = f"{interval} min"
                    ws_summary[f'B{row}'] = stats.total_count
                    ws_summary[f'C{row}'] = f"{stats.flow_rate_per_hour:.1f}"
                    ws_summary[f'D{row}'] = stats.direction_ab
                    ws_summary[f'E{row}'] = stats.direction_ba
                    row += 1
            
            row += 1
        
        # Sheet 2: Events
        ws_events = wb.create_sheet("EVENTOS", 1)
        
        ws_events['A1'] = "EVENTOS DE CRUCE"
        ws_events['A1'].font = title_font
        
        headers = ["Hora", "ID Track", "Tipo", "Confianza", "Dirección", "Línea", "Posición X", "Posición Y"]
        for col, header in enumerate(headers, 1):
            cell = ws_events.cell(row=3, column=col)
            cell.value = header
            cell.fill = header_fill
            cell.font = header_font
        
        event_row = 4
        for event in lines_manager.get_events(last_n=500):
            line = lines_manager.get_line(event.line_id)
            line_name = line.name if line else "N/A"
            
            ws_events.cell(row=event_row, column=1).value = event.timestamp.strftime("%H:%M:%S")
            ws_events.cell(row=event_row, column=2).value = event.track_id
            ws_events.cell(row=event_row, column=3).value = event.class_label
            ws_events.cell(row=event_row, column=4).value = f"{event.confidence:.2%}"
            ws_events.cell(row=event_row, column=5).value = event.direction
            ws_events.cell(row=event_row, column=6).value = line_name
            ws_events.cell(row=event_row, column=7).value = f"{event.position_x:.2f}"
            ws_events.cell(row=event_row, column=8).value = f"{event.position_y:.2f}"
            
            event_row += 1
        
        # Auto-width columns
        for ws in [ws_summary, ws_events]:
            for column in ws.columns:
                max_length = 0
                column_letter = get_column_letter(column[0].column)
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                ws.column_dimensions[column_letter].width = adjusted_width
        
        # Save
        excel_path = self.export_dir / f"session_{timestamp}.xlsx"
        wb.save(excel_path)
        
        return excel_path
    
    def _export_csv(self, session_data: Dict, lines_manager, timestamp: str) -> Path:
        """Export events to CSV"""
        csv_path = self.export_dir / f"events_{timestamp}.csv"
        
        fieldnames = [
            'timestamp', 'track_id', 'class_label', 'confidence',
            'direction', 'line_name', 'position_x', 'position_y'
        ]
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for event in lines_manager.get_events(last_n=5000):
                line = lines_manager.get_line(event.line_id)
                line_name = line.name if line else "N/A"
                
                writer.writerow({
                    'timestamp': event.timestamp.isoformat(),
                    'track_id': event.track_id,
                    'class_label': event.class_label,
                    'confidence': f"{event.confidence:.2%}",
                    'direction': event.direction,
                    'line_name': line_name,
                    'position_x': f"{event.position_x:.4f}",
                    'position_y': f"{event.position_y:.4f}",
                })
        
        return csv_path


class SessionManager:
    """Manage traffic monitoring sessions"""
    
    def __init__(self, sessions_dir: str = "sessions"):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.current_session = None
    
    def create_session(self, name: str, location: str = "", observer: str = "",
                      description: str = "", direction_a: str = "A",
                      direction_b: str = "B") -> Dict:
        """Create a new monitoring session"""
        session = {
            'id': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'name': name,
            'location': location,
            'observer': observer,
            'description': description,
            'direction_a': direction_a,
            'direction_b': direction_b,
            'start_time': datetime.now().isoformat(),
            'events_count': 0,
        }
        
        self.current_session = session
        self._save_session_meta(session)
        
        return session
    
    def close_session(self) -> bool:
        """Close current session"""
        if self.current_session:
            self.current_session['end_time'] = datetime.now().isoformat()
            self._save_session_meta(self.current_session)
            self.current_session = None
            return True
        return False
    
    def get_current_session(self) -> Optional[Dict]:
        """Get active session"""
        return self.current_session
    
    def _save_session_meta(self, session: Dict):
        """Save session metadata to file"""
        meta_path = self.sessions_dir / f"{session['id']}_meta.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            # Convert datetime to strings for JSON serialization
            session_copy = session.copy()
            json.dump(session_copy, f, indent=2, ensure_ascii=False)


# Export functions for use in GUI
def export_traffic_report(session_data: Dict, lines_manager, format: str = "both") -> Dict[str, Path]:
    """Quick export function"""
    exporter = TrafficExporter()
    return exporter.export_session(session_data, lines_manager, format)


def create_new_session(name: str, **kwargs) -> Dict:
    """Quick session creation function"""
    manager = SessionManager()
    return manager.create_session(name, **kwargs)

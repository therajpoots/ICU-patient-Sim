"""
Report Selection and Generation Dialog Box.
Allows doctors to select start/end datetimes, view logged anomalies from the database,
configure patient demographic metadata, and trigger PDF clinical report compilation.
"""

import os
import time
from datetime import datetime, timedelta
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit, QDateTimeEdit,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox, QFrame
)
from PyQt6.QtCore import Qt, QDateTime, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor

from gui.database import EncryptedDatabase
from gui.pdf_generator import generate_pdf_report

class PDFCompileWorker(QThread):
    finished = pyqtSignal(bool, str) # success, message
    
    def __init__(self, patient_info, anomaly_logs, output_path, api_key):
        super().__init__()
        self.patient_info = patient_info
        self.anomaly_logs = anomaly_logs
        self.output_path = output_path
        self.api_key = api_key
        
    def run(self):
        try:
            generate_pdf_report(
                patient_info=self.patient_info,
                anomaly_logs=self.anomaly_logs,
                output_path=self.output_path,
                api_key=self.api_key
            )
            self.finished.emit(True, f"Clinical Report successfully compiled and saved to:\n{self.output_path}")
        except Exception as e:
            self.finished.emit(False, str(e))

class ReportDialog(QDialog):
    def __init__(self, password: str, api_key: str = "sk-3ae47177f18e4ecf808440d6168c0d6f", patient_info: dict = None, parent=None):
        super().__init__(parent)
        self.password = password
        self.api_key = api_key
        self.patient_info = patient_info
        self.db = EncryptedDatabase()
        
        self.setWindowTitle("Generate Clinical Telemetry Report")
        self.setMinimumSize(680, 500)
        self.setObjectName("ReportDialog")
        
        self.init_ui()
        self.load_data()

    def init_ui(self):
        # Dialog-wide stylesheet (matching app dark mode)
        self.setStyleSheet("""
            QDialog#ReportDialog {
                background-color: #121214;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
            }
            QLabel#DialogTitle {
                color: #7c3aed; /* Purple brand color */
                font-size: 16px;
                font-weight: bold;
            }
            QFrame#CardFrame {
                background-color: #1a1a1f;
                border: 1px solid #2d2d34;
                border-radius: 6px;
            }
            QLineEdit {
                background-color: #121214;
                border: 1px solid #2d2d34;
                color: #ffffff;
                padding: 5px;
                border-radius: 4px;
            }
            QDateTimeEdit {
                background-color: #121214;
                border: 1px solid #2d2d34;
                color: #ffffff;
                padding: 4px;
                border-radius: 4px;
            }
            QTableWidget {
                background-color: #121214;
                border: 1px solid #2d2d34;
                color: #ffffff;
                gridline-color: #2d2d34;
            }
            QHeaderView::section {
                background-color: #1a1a1f;
                color: #7c3aed;
                padding: 4px;
                border: 1px solid #2d2d34;
                font-weight: bold;
            }
            QPushButton#ActionBtn {
                background-color: #7c3aed;
                color: #ffffff;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton#ActionBtn:hover {
                background-color: #8b5cf6;
            }
            QPushButton#SecBtn {
                background-color: #27272a;
                color: #ffffff;
                border: 1px solid #2d2d34;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QPushButton#SecBtn:hover {
                background-color: #3f3f46;
            }
        """)

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Header Title
        title_label = QLabel("ICU CLINICAL CASE REPORT DESIGNER")
        title_label.setObjectName("DialogTitle")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Section A: Patient Profile Card
        patient_card = QFrame()
        patient_card.setObjectName("CardFrame")
        patient_layout = QGridLayout(patient_card)
        patient_layout.setContentsMargins(15, 15, 15, 15)
        patient_layout.setSpacing(10)
        
        patient_layout.addWidget(QLabel("<b>Patient Name:</b>"), 0, 0)
        default_name = self.patient_info.get("name", "Rana Talha Khalid") if self.patient_info else "Rana Talha Khalid"
        self.name_input = QLineEdit(default_name)
        patient_layout.addWidget(self.name_input, 0, 1)
        
        patient_layout.addWidget(QLabel("<b>Patient ID:</b>"), 0, 2)
        default_id = self.patient_info.get("id", "PT-2026-9041") if self.patient_info else "PT-2026-9041"
        self.id_input = QLineEdit(default_id)
        patient_layout.addWidget(self.id_input, 0, 3)
        
        patient_layout.addWidget(QLabel("<b>Ward / Bed:</b>"), 1, 0)
        if self.patient_info:
            default_ward = f"{self.patient_info.get('ward', 'ICU')} - {self.patient_info.get('bed', 'Bed 04')}"
        else:
            default_ward = "ICU - Bed 04"
        self.ward_input = QLineEdit(default_ward)
        patient_layout.addWidget(self.ward_input, 1, 1)
        
        patient_layout.addWidget(QLabel("<b>Patient Age:</b>"), 1, 2)
        default_age = self.patient_info.get("age", "29") if self.patient_info else "29"
        self.age_input = QLineEdit(default_age)
        patient_layout.addWidget(self.age_input, 1, 3)
        
        layout.addWidget(patient_card)
        
        # Section B: Datetime Selectors
        time_card = QFrame()
        time_card.setObjectName("CardFrame")
        time_layout = QHBoxLayout(time_card)
        time_layout.setContentsMargins(15, 10, 15, 10)
        time_layout.setSpacing(10)
        
        time_layout.addWidget(QLabel("<b>Start Time:</b>"))
        self.start_dt = QDateTimeEdit(QDateTime.currentDateTime().addSecs(-3600))
        self.start_dt.setCalendarPopup(True)
        time_layout.addWidget(self.start_dt)
        
        time_layout.addWidget(QLabel("<b>End Time:</b>"))
        self.end_dt = QDateTimeEdit(QDateTime.currentDateTime())
        self.end_dt.setCalendarPopup(True)
        time_layout.addWidget(self.end_dt)
        
        self.refresh_btn = QPushButton("Filter Logs")
        self.refresh_btn.setObjectName("SecBtn")
        self.refresh_btn.clicked.connect(self.load_data)
        time_layout.addWidget(self.refresh_btn)
        
        # Quick Time presets
        self.last_hour_btn = QPushButton("Last 1 hr")
        self.last_hour_btn.setObjectName("SecBtn")
        self.last_hour_btn.clicked.connect(self.set_last_hour)
        time_layout.addWidget(self.last_hour_btn)
        
        layout.addWidget(time_card)
        
        # Section C: Logs Table View (Anomalies to include)
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setRowCount(0)
        self.table.setHorizontalHeaderLabels(["Timestamp", "Duration", "Anomaly State", "Peak HR / BP"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table)
        
        # Action row
        bottom_layout = QHBoxLayout()
        self.cancel_btn = QPushButton("Close")
        self.cancel_btn.setObjectName("SecBtn")
        self.cancel_btn.clicked.connect(self.reject)
        
        self.compile_btn = QPushButton("Compile PDF Report")
        self.compile_btn.setObjectName("ActionBtn")
        self.compile_btn.clicked.connect(self.compile_report)
        
        bottom_layout.addWidget(self.cancel_btn)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.compile_btn)
        layout.addLayout(bottom_layout)
        
        self.setLayout(layout)

    def set_last_hour(self):
        """Sets the filter datetime to cover the last hour."""
        now = QDateTime.currentDateTime()
        self.start_dt.setDateTime(now.addSecs(-3600))
        self.end_dt.setDateTime(now)
        self.load_data()

    def load_data(self):
        """Loads and filters anomaly events from encrypted database."""
        start_ts = self.start_dt.dateTime().toSecsSinceEpoch()
        end_ts = self.end_dt.dateTime().toSecsSinceEpoch()
        
        try:
            logs = self.db.get_anomaly_logs(
                password=self.password,
                start_time=start_ts,
                end_time=end_ts
            )
            self.loaded_logs = logs
            self.update_table(logs)
        except Exception as e:
            QMessageBox.critical(self, "Database Error", f"Failed to decrypt database logs:\n{e}")
            self.loaded_logs = []

    def update_table(self, logs):
        """Updates table rows with log values."""
        self.table.setRowCount(len(logs))
        for row, log in enumerate(logs):
            dt_str = datetime.fromtimestamp(log.get("start_time", time.time())).strftime('%Y-%m-%d %H:%M:%S')
            
            start_t = log.get("start_time", 0.0)
            end_t = log.get("end_time", 0.0)
            duration = f"{end_t - start_t:.0f}s"
            
            state = log.get("state", "unknown").replace("_", " ").title()
            
            v = log.get("vitals", {})
            if not isinstance(v, dict):
                v = {}
            hr = v.get("heart_rate", 0.0)
            sbp = v.get("systolic_bp", 0.0)
            dbp = v.get("diastolic_bp", 0.0)
            vitals_str = f"HR:{hr:.0f} | BP:{sbp:.0f}/{dbp:.0f}"
            
            self.table.setItem(row, 0, QTableWidgetItem(dt_str))
            self.table.setItem(row, 1, QTableWidgetItem(duration))
            self.table.setItem(row, 2, QTableWidgetItem(state))
            self.table.setItem(row, 3, QTableWidgetItem(vitals_str))

    def compile_report(self):
        """Validates logs and opens PDF file dialog to compile report."""
        if not self.loaded_logs:
            QMessageBox.warning(self, "Compilation Error", "No anomaly records found in the selected range to include in report.")
            return
            
        patient_info = {
            "patient_name": self.name_input.text().strip(),
            "patient_id": self.id_input.text().strip(),
            "ward": self.ward_input.text().strip(),
            "patient_age": self.age_input.text().strip()
        }
        
        # Save file dialog
        default_name = f"clinical_report_{patient_info['patient_name'].replace(' ', '_')}_{int(time.time())}.pdf"
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Patient Telemetry Report",
            default_name,
            "PDF Files (*.pdf)"
        )
        
        if not output_path:
            return
            
        # Compile PDF in background thread
        self.compile_btn.setEnabled(False)
        self.compile_btn.setText("Generating PDF...")
        self.repaint()
        
        self.pdf_worker = PDFCompileWorker(patient_info, self.loaded_logs, output_path, self.api_key)
        self.pdf_worker.finished.connect(self.on_pdf_compiled)
        self.pdf_worker.start()

    def on_pdf_compiled(self, success, message):
        self.compile_btn.setEnabled(True)
        self.compile_btn.setText("Compile PDF Report")
        if success:
            QMessageBox.information(self, "Success", message)
            self.accept()
        else:
            QMessageBox.critical(self, "Compilation Failed", f"An error occurred during report generation:\n{message}")

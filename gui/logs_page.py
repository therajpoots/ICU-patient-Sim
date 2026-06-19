"""
Historical Anomaly Logs Page.
Queries and lists decrypted anomaly logs from the database, allows filtering,
and triggers the clinical report dialogue or record deletions.
"""

from datetime import datetime
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QTableWidget, QTableWidgetItem, QHeaderView, QFrame, QMessageBox, QDateTimeEdit
)
from PyQt6.QtCore import pyqtSignal, Qt, QDateTime
from PyQt6.QtGui import QFont, QColor

from gui.database import EncryptedDatabase

class LogsPage(QWidget):
    # Signals
    back_requested = pyqtSignal()
    report_requested = pyqtSignal()

    def __init__(self, password: str = ""):
        super().__init__()
        self.password = password
        self.db = EncryptedDatabase()
        self.setObjectName("LogsPage")
        self.init_ui()

    def set_password(self, password: str):
        """Sets the decryption password dynamically from connection setup."""
        self.password = password
        self.load_logs()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Header block
        header_layout = QHBoxLayout()
        title = QLabel("PATIENT ANOMALY TELEMETRY ARCHIVE")
        title.setObjectName("LogsPageTitle")
        title.setFont(QFont("Outfit", 16, QFont.Weight.Bold))
        title.setStyleSheet("color: #7c3aed;")
        
        self.back_btn = QPushButton("← Back to Monitor")
        self.back_btn.setObjectName("SecBtn")
        self.back_btn.clicked.connect(self.back_requested.emit)
        
        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(self.back_btn)
        layout.addLayout(header_layout)
        
        # Filtering toolbar
        filter_frame = QFrame()
        filter_frame.setObjectName("FilterFrame")
        filter_layout = QHBoxLayout(filter_frame)
        filter_layout.setContentsMargins(15, 10, 15, 10)
        filter_layout.setSpacing(10)
        
        filter_layout.addWidget(QLabel("<b>Start Time:</b>"))
        self.start_dt = QDateTimeEdit(QDateTime.currentDateTime().addDays(-1))
        self.start_dt.setCalendarPopup(True)
        filter_layout.addWidget(self.start_dt)
        
        filter_layout.addWidget(QLabel("<b>End Time:</b>"))
        self.end_dt = QDateTimeEdit(QDateTime.currentDateTime())
        self.end_dt.setCalendarPopup(True)
        filter_layout.addWidget(self.end_dt)
        
        self.filter_btn = QPushButton("Filter Logs")
        self.filter_btn.setObjectName("SecBtn")
        self.filter_btn.clicked.connect(self.load_logs)
        filter_layout.addWidget(self.filter_btn)
        
        self.clear_filter_btn = QPushButton("Show All")
        self.clear_filter_btn.setObjectName("SecBtn")
        self.clear_filter_btn.clicked.connect(self.clear_filter)
        filter_layout.addWidget(self.clear_filter_btn)
        
        layout.addWidget(filter_frame)
        
        # Table widget
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["ID", "Timestamp", "Duration", "arrhythmia / State", "Average Peak Vitals"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setObjectName("LogsTable")
        layout.addWidget(self.table)
        
        # Table Style
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #1a1a1f;
                border: 1px solid #2d2d34;
                color: #ffffff;
                gridline-color: #2d2d34;
            }
            QHeaderView::section {
                background-color: #121214;
                color: #7c3aed;
                padding: 6px;
                border: 1px solid #2d2d34;
                font-weight: bold;
            }
        """)
        
        # Action Footer
        action_layout = QHBoxLayout()
        self.delete_btn = QPushButton("Delete Selected Log")
        self.delete_btn.setObjectName("SecBtn")
        self.delete_btn.setStyleSheet("color: #ef4444; border-color: #ef4444;")
        self.delete_btn.clicked.connect(self.delete_selected_log)
        
        self.report_btn = QPushButton("Generate Clinical Report Document")
        self.report_btn.setObjectName("ActionBtn")
        self.report_btn.setStyleSheet("""
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
        """)
        self.report_btn.clicked.connect(self.report_requested.emit)
        
        action_layout.addWidget(self.delete_btn)
        action_layout.addStretch()
        action_layout.addWidget(self.report_btn)
        layout.addLayout(action_layout)
        
        self.setLayout(layout)

    def clear_filter(self):
        """Resets filters to covers last 7 days and reloads."""
        now = QDateTime.currentDateTime()
        self.start_dt.setDateTime(now.addDays(-7))
        self.end_dt.setDateTime(now)
        self.load_logs()

    def load_logs(self):
        """Loads anomaly logs within date ranges and populates table."""
        if not self.password:
            return
            
        start_ts = self.start_dt.dateTime().toSecsSinceEpoch()
        end_ts = self.end_dt.dateTime().toSecsSinceEpoch()
        
        try:
            logs = self.db.get_anomaly_logs(
                password=self.password,
                start_time=start_ts,
                end_time=end_ts
            )
            self.populate_table(logs)
        except Exception as e:
            # Silence error or show validation alert
            print(f"Decryption failed during log list load: {e}")

    def populate_table(self, logs):
        """Draws rows on logs table."""
        self.table.setRowCount(len(logs))
        for row, log in enumerate(logs):
            dt_str = datetime.fromtimestamp(log["start_time"]).strftime('%Y-%m-%d %H:%M:%S')
            duration = f"{log['end_time'] - log['start_time']:.0f} seconds"
            state = log["state"].replace("_", " ").title()
            
            v = log["vitals"]
            vitals_str = f"HR: {v['heart_rate']:.0f} bpm | BP: {v['systolic_bp']:.0f}/{v['diastolic_bp']:.0f} mmHg | SpO₂: {v['spo2']:.0f}%"
            
            id_item = QTableWidgetItem(str(log["id"]))
            id_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            id_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            
            dt_item = QTableWidgetItem(dt_str)
            dt_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            
            dur_item = QTableWidgetItem(duration)
            dur_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            
            state_item = QTableWidgetItem(state)
            state_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            state_item.setForeground(QColor("#ff5c8a") if log["state"] in ("vfib", "ventricular_fibrillation") else QColor("#ffffff"))
            
            v_item = QTableWidgetItem(vitals_str)
            v_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            
            self.table.setItem(row, 0, id_item)
            self.table.setItem(row, 1, dt_item)
            self.table.setItem(row, 2, dur_item)
            self.table.setItem(row, 3, state_item)
            self.table.setItem(row, 4, v_item)

    def delete_selected_log(self):
        """Deletes a selected log row from SQLite database."""
        selected_rows = self.table.selectedItems()
        if not selected_rows:
            QMessageBox.warning(self, "Selection Error", "Please select a log to delete.")
            return
            
        row = selected_rows[0].row()
        log_id = int(self.table.item(row, 0).text())
        
        reply = QMessageBox.question(
            self, "Confirm Delete", 
            f"Are you sure you want to permanently delete Log ID {log_id} from the encrypted archive?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                import sqlite3
                conn = sqlite3.connect(self.db.db_path)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM anomaly_logs WHERE id = ?", (log_id,))
                conn.commit()
                conn.close()
                QMessageBox.information(self, "Success", f"Log ID {log_id} successfully deleted.")
                self.load_logs()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete record from DB: {e}")

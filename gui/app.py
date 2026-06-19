"""
Main Application Coordinator for Antigravity Patient Monitor.
Integrates stacked UI pages, worker threads, database logic, and background MCP subprocesses.
"""

import os
import sys
import subprocess
from PyQt6.QtWidgets import QMainWindow, QStackedWidget, QWidget, QVBoxLayout, QMessageBox, QApplication, QDialog
from PyQt6.QtCore import Qt

from gui.connection_page import ConnectionPage
from gui.monitor_page import MonitorPage
from gui.logs_page import LogsPage
from gui.features_page import FeaturesPage
from gui.report_dialog import ReportDialog
from gui.worker import MonitoringWorker

class PatientMonitorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HealthFi - Bedside Telemetry Monitor")
        self.setObjectName("MainWindow")
        
        # State variables
        self.worker = None
        self.mcp_proc = None
        self.password = ""
        self.api_key = ""
        
        # Sizing and framing: 720p compatibility
        self.setMinimumSize(960, 600)
        self.resize(1100, 660)
        self.center_on_screen()
        
        self.init_ui()
        self.apply_theme()

    def center_on_screen(self):
        """Centers the application window on the active monitor screen."""
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            x = (geo.width() - self.width()) // 2
            y = (geo.height() - self.height()) // 2
            # Lift slightly above perfect vertical center for aesthetic alignment
            y = max(0, y - 25)
            self.move(x, y)

    def init_ui(self):
        # Stacked widgets to handle transitions (Index 0: Monitor, Index 1: Logs, Index 2: Features)
        self.stacked_widget = QStackedWidget()
        
        # Instantiate pages
        self.monitor_page = MonitorPage()
        self.logs_page = LogsPage()
        self.features_page = FeaturesPage()
        
        # Add to stack
        self.stacked_widget.addWidget(self.monitor_page) # Index 0
        self.stacked_widget.addWidget(self.logs_page) # Index 1
        self.stacked_widget.addWidget(self.features_page) # Index 2
        
        # Connect signals
        self.monitor_page.stop_requested.connect(self.stop_monitoring)
        self.monitor_page.inject_anomaly_requested.connect(self.inject_anomaly)
        self.monitor_page.report_requested.connect(self.show_report_dialog)
        self.monitor_page.view_logs_requested.connect(self.show_logs_page)
        self.monitor_page.view_features_requested.connect(self.show_features_page)
        self.monitor_page.emergency_triggered.connect(self.log_emergency_call)
        self.monitor_page.patient_info_changed.connect(self.features_page.update_patient_info)
        
        self.logs_page.back_requested.connect(self.show_monitor_page)
        self.logs_page.report_requested.connect(self.show_report_dialog)
        
        self.features_page.stop_requested.connect(self.stop_monitoring)
        self.features_page.view_live_requested.connect(self.show_monitor_page)
        self.features_page.report_requested.connect(self.show_report_dialog)
        self.features_page.view_logs_requested.connect(self.show_logs_page)
        
        self.setCentralWidget(self.stacked_widget)
        self.stacked_widget.setCurrentIndex(0)

    def start_monitoring(self, mode, host, port, api_key, password, mcp_enabled):
        """Starts background worker thread and optionally launches local MCP server."""
        self.password = password
        self.api_key = api_key
        
        # Initialize logs page password
        self.logs_page.set_password(password)
        
        # 1. Instantiate background worker thread
        self.worker = MonitoringWorker(
            mode=mode, 
            host=host, 
            port=port, 
            api_key=api_key, 
            password=password
        )
        
        # Connect worker signals to monitor UI
        self.worker.vitals_updated.connect(self.monitor_page.update_vitals)
        self.worker.vitals_updated.connect(self.features_page.update_vitals)
        self.worker.waveforms_updated.connect(self.monitor_page.update_waveforms)
        self.worker.anomaly_logged.connect(self.monitor_page.add_recent_anomaly)
        self.worker.demo_timer_updated.connect(self.monitor_page.update_demo_countdown)
        self.worker.demo_timer_updated.connect(self.features_page.update_demo_countdown)
        self.worker.error_occurred.connect(self.handle_worker_error)
        self.worker.finished_monitoring.connect(self.on_worker_finished)
        
        # Set UI labels depending on mode
        if mode == "demo":
            self.monitor_page.update_demo_countdown(600)
            self.features_page.update_demo_countdown(600)
        else:
            self.monitor_page.set_active_monitoring_label()
            self.features_page.set_active_monitoring_label()
            
        # 2. Launch local MCP server in background if requested
        if mcp_enabled:
            self.start_mcp_server(mode, host, port, api_key, password)
            
        # 3. Start worker and transition to monitor view
        self.worker.start()
        self.stacked_widget.setCurrentIndex(0)

    def stop_monitoring(self):
        """Stops thread loops and terminates subprocesses, switching back to setup dialog."""
        # Stop all plotter timers first to prevent use-after-free crashes
        self.monitor_page.cleanup()

        if self.worker:
            try:
                self.worker.disconnect()
            except Exception:
                pass
            self.worker.stop()
            self.worker.wait(1000)
            self.worker = None

        self.stop_mcp_server()
        
        self.hide()
        dialog = ConnectionPage()
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.start_monitoring(
                mode=dialog.selected_mode,
                host=dialog.selected_host,
                port=dialog.selected_port,
                api_key=dialog.selected_api_key,
                password=dialog.selected_password,
                mcp_enabled=dialog.selected_mcp_enabled
            )
            self.show()
        else:
            self.close()

    def inject_anomaly(self, state):
        """Injects anomaly via worker's trigger routine."""
        if self.worker:
            self.worker.trigger_anomaly(state)

    def log_emergency_call(self):
        """Encrypts and logs a simulated clinician nurse call event to the database."""
        from gui.database import EncryptedDatabase
        import time
        db = EncryptedDatabase()
        try:
            db.log_anomaly(
                password=self.password,
                start_time=time.time(),
                end_time=time.time() + 15.0,
                state="nurse_call",
                vitals={
                    "heart_rate": 80.0,
                    "systolic_bp": 120.0,
                    "diastolic_bp": 80.0,
                    "spo2": 98.0,
                    "respiratory_rate": 16.0,
                    "core_temperature": 37.0,
                    "skin_temperature": 35.5
                },
                waveforms={"ecg": [], "ppg": [], "rsp": []}
            )
            print("Logged nurse call event in database.")
        except Exception as e:
            print(f"Failed to log nurse call: {e}")

    def show_report_dialog(self):
        """Opens the report design modal dialogue."""
        patient_info = {
            "name": self.monitor_page.patient_name,
            "id": self.monitor_page.patient_id,
            "ward": self.monitor_page.patient_ward,
            "unit": self.monitor_page.patient_unit,
            "bed": self.monitor_page.patient_bed,
            "age": self.monitor_page.patient_age
        }
        dialog = ReportDialog(self.password, self.api_key, patient_info, self)
        dialog.exec()

    def show_logs_page(self):
        """Navigates stack to Anomaly Archive view."""
        self.logs_page.load_logs()
        self.stacked_widget.setCurrentIndex(1)

    def show_features_page(self):
        """Navigates stack to Tabular Features view."""
        self.stacked_widget.setCurrentIndex(2)

    def show_monitor_page(self):
        """Navigates stack back to Realtime Monitor view."""
        self.stacked_widget.setCurrentIndex(0)

    def handle_worker_error(self, err_msg):
        """Logs worker communication or decryption errors."""
        print(f"Worker Error: {err_msg}")
        # Note: We avoid popping up dialogue boxes on every second's SSE timeout to avoid UI spam.
        # Errors will be printed to terminal console or status log.

    def on_worker_finished(self):
        """Fires when worker loop completes (e.g. demo mode completed)."""
        QMessageBox.information(self, "Session Ended", "Monitoring session has completed successfully.")
        self.stop_monitoring()

    def start_mcp_server(self, mode, host, port, api_key, password):
        """Launches mcp_server.py as a background process with environmental configurations."""
        self.stop_mcp_server()
        
        env = os.environ.copy()
        env["DB_PASSWORD"] = password
        env["API_KEY"] = api_key
        env["API_HOST"] = host
        env["API_PORT"] = str(port)
        env["MONITORING_MODE"] = mode
        
        try:
            # We run python mcp_server.py in background
            self.mcp_proc = subprocess.Popen(
                [sys.executable, "mcp_server.py"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print("INFO: Local MCP Server started successfully in background.")
        except Exception as e:
            print(f"ERROR: Failed to launch MCP Server background process: {e}")

    def stop_mcp_server(self):
        """Terminates the background MCP server process if running."""
        if self.mcp_proc:
            try:
                self.mcp_proc.terminate()
                self.mcp_proc.wait(timeout=2.0)
                print("INFO: MCP Server background process terminated.")
            except Exception:
                try:
                    self.mcp_proc.kill()
                    print("INFO: MCP Server process force-killed.")
                except Exception:
                    pass
            self.mcp_proc = None

    def closeEvent(self, event):
        """Ensures all timers, threads, and servers clean up on close."""
        self.monitor_page.cleanup()
        self.stop_monitoring()
        event.accept()

    def apply_theme(self):
        """Applies high-fidelity dark-neon cyberpunk medical stylesheet."""
        self.setStyleSheet("""
            /* Base Theme Configuration */
            QMainWindow {
                background-color: #0b0b0d;
            }
            QWidget {
                color: #e2e8f0;
                font-family: 'Inter', 'Outfit', sans-serif;
            }
            
            /* Frames & Cards */
            QFrame#FormCard {
                background-color: #121217;
                border: 1px solid #27272a;
                border-radius: 12px;
            }
            QFrame#ChartsFrame, QFrame#RightControlFrame, QFrame#FilterFrame, QFrame#ControlsSubFrame {
                background-color: #101014;
                border: 1px solid #1f1f23;
                border-radius: 8px;
            }
            QFrame#VitalsFrame {
                background-color: #101014;
                border: 1px solid #27272a;
                border-radius: 8px;
            }
            
            /* Neon Telemetry Readouts */
            QFrame#VitalBox_HR {
                background-color: #141418;
                border: 1px solid #10b981; /* Neon green */
                border-radius: 8px;
            }
            QFrame#VitalBox_BP {
                background-color: #141418;
                border: 1px solid #f43f5e; /* Neon Pink/BP */
                border-radius: 8px;
            }
            QFrame#VitalBox_SpO2 {
                background-color: #141418;
                border: 1px solid #06b6d4; /* Neon Cyan */
                border-radius: 8px;
            }
            QFrame#VitalBox_RR {
                background-color: #141418;
                border: 1px solid #f59e0b; /* Neon Amber */
                border-radius: 8px;
            }
            
            /* Input Widgets */
            QLineEdit, QComboBox {
                background-color: #191920;
                border: 1px solid #27272a;
                border-radius: 6px;
                padding: 6px 12px;
                color: #ffffff;
                font-size: 11px;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 1px solid #7c3aed; /* Purple brand glow */
                background-color: #1b1b24;
            }
            
            /* ComboBox Dropdown popup styling */
            QComboBox QAbstractItemView {
                background-color: #121217;
                border: 1px solid #27272a;
                selection-background-color: #7c3aed;
                selection-color: #ffffff;
                color: #e2e8f0;
            }
            
            QCheckBox {
                color: #94a3b8;
            }
            
            /* ScrollBar Customization */
            QScrollBar:vertical {
                border: none;
                background: #0f0f13;
                width: 8px;
                margin: 0px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #2d2d34;
                min-height: 20px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: #3f3f46;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            
            /* Form Titles / Labels */
            QLabel#MainTitle {
                color: #ffffff;
                letter-spacing: 1.5px;
            }
            QLabel#SubTitle {
                color: #71717a;
            }
            QLabel#FormLabel {
                color: #a1a1aa;
            }
            QLabel#VitalsPanelTitle, QLabel#FeaturesPanelTitle, QLabel#LogsPageTitle {
                color: #7c3aed; /* Purple brand accent */
                letter-spacing: 0.8px;
            }
            
            /* Action Buttons (Sexy styling) */
            QPushButton#StartBtn {
                background-color: #10b981; /* Neon green */
                color: #0b0b0d;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton#StartBtn:hover {
                background-color: #34d399;
            }
            QPushButton#TestBtn {
                background-color: #06b6d4; /* Cyan */
                color: #0b0b0d;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton#TestBtn:hover {
                background-color: #22d3ee;
            }
            QPushButton#GenerateReportBtn {
                background-color: #7c3aed; /* Purple primary */
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton#GenerateReportBtn:hover {
                background-color: #8b5cf6;
            }
            QPushButton#ArchiveBtn {
                background-color: #1f1f23;
                color: #ffffff;
                border: 1px solid #2d2d34;
                border-radius: 6px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton#ArchiveBtn:hover {
                background-color: #2d2d34;
                border-color: #3f3f46;
            }
            QPushButton#StopBtn {
                background-color: #16161a;
                color: #ef4444;
                border: 1px solid #ef4444;
                border-radius: 6px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton#StopBtn:hover {
                background-color: #ef4444;
                color: #ffffff;
            }
            QPushButton#InjectBtn {
                background-color: #e11d48; /* Coral Red alert trigger */
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton#InjectBtn:hover {
                background-color: #f43f5e;
            }
            QPushButton#SecBtn {
                background-color: #1f1f23;
                color: #ffffff;
                border: 1px solid #2d2d34;
                border-radius: 6px;
                padding: 6px 12px;
            }
            QPushButton#SecBtn:hover {
                background-color: #2d2d34;
            }
            
            /* Tables flat modern theme */
            QTableWidget {
                background-color: #101014;
                border: 1px solid #1f1f23;
                color: #e2e8f0;
                gridline-color: #1a1a1f;
                border-radius: 6px;
            }
            QHeaderView::section {
                background-color: #16161a;
                color: #a0aec0;
                padding: 6px;
                border: 1px solid #1f1f23;
                font-weight: bold;
            }
        """)

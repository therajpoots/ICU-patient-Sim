"""
Tabular Features View Page - Glassmorphic Premium Redesign.
Displays the live 39 extracted clinical features grouped by category,
showing their expected baseline bounds, drift status, and SHAP feature influence.
"""

import time
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QTableWidget, QTableWidgetItem, QHeaderView,
    QScrollArea, QProgressBar, QSizePolicy
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QFont, QColor

class FeaturesPage(QWidget):
    stop_requested = pyqtSignal()
    view_live_requested = pyqtSignal()
    report_requested = pyqtSignal()
    view_logs_requested = pyqtSignal()

    # Define feature details: (key, category, label, expected_range_str, format_func, check_drift_func)
    # Check drift return: (status_text, color_hex)
    FEATURE_DEFS = [
        # Temperature
        ("temp_core", "Temperature", "Core Temperature", "36.5 - 37.5 °C", 
         lambda v: f"{v:.2f} °C", 
         lambda v: ("Critical", "#ef4444") if (v < 35.0 or v > 38.5) else (("Elevated", "#f59e0b") if v > 37.5 else (("Reduced", "#f59e0b") if v < 36.5 else ("Normal", "#10b981")))),
        ("temp_skin", "Temperature", "Skin Temperature", "33.0 - 36.0 °C", 
         lambda v: f"{v:.2f} °C", 
         lambda v: ("Elevated", "#f59e0b") if v > 36.0 else (("Reduced", "#f59e0b") if v < 33.0 else ("Normal", "#10b981"))),
        
        # Blood Pressure
        ("bp_sbp", "Blood Pressure", "Systolic BP", "90 - 140 mmHg", 
         lambda v: f"{v:.1f} mmHg", 
         lambda v: ("Critical", "#ef4444") if (v < 80 or v > 165) else (("Elevated", "#f59e0b") if v > 140 else (("Reduced", "#f59e0b") if v < 90 else ("Normal", "#10b981")))),
        ("bp_dbp", "Blood Pressure", "Diastolic BP", "60 - 90 mmHg", 
         lambda v: f"{v:.1f} mmHg", 
         lambda v: ("Elevated", "#f59e0b") if v > 90 else (("Reduced", "#f59e0b") if v < 60 else ("Normal", "#10b981"))),
        ("bp_map", "Blood Pressure", "Mean Arterial Pressure (MAP)", "70 - 105 mmHg", 
         lambda v: f"{v:.1f} mmHg", 
         lambda v: ("Critical", "#ef4444") if v < 60 else (("Elevated", "#f59e0b") if v > 105 else (("Reduced", "#f59e0b") if v < 70 else ("Normal", "#10b981")))),
        ("bp_pulse_pressure", "Blood Pressure", "Pulse Pressure", "30 - 50 mmHg", 
         lambda v: f"{v:.1f} mmHg", 
         lambda v: ("Elevated", "#f59e0b") if (v > 50 or v < 30) else ("Normal", "#10b981")),
        
        # ECG Features
        ("ecg_hr", "ECG / Cardiac", "ECG Heart Rate", "60 - 100 bpm", 
         lambda v: f"{v:.1f} bpm", 
         lambda v: ("Critical", "#ef4444") if (v < 45 or v > 130) else (("Elevated", "#f59e0b") if v > 100 else (("Reduced", "#f59e0b") if v < 60 else ("Normal", "#10b981")))),
        ("ecg_qrs_duration", "ECG / Cardiac", "QRS Duration", "70 - 110 ms", 
         lambda v: f"{v * 1000:.0f} ms" if v < 1.0 else f"{v:.0f} ms", 
         lambda v: ("Critical", "#ef4444") if (v > 0.12 if v < 1.0 else v > 120) else ("Normal", "#10b981")),
        ("ecg_st_level", "ECG / Cardiac", "ST Segment Level", "-0.10 - 0.10 mV", 
         lambda v: f"{v:+.3f} mV", 
         lambda v: ("Critical", "#ef4444") if (v > 0.20 or v < -0.15) else (("Elevated", "#f59e0b") if v > 0.10 else (("Reduced", "#f59e0b") if v < -0.10 else ("Normal", "#10b981")))),
        ("ecg_pr_interval", "ECG / Cardiac", "PR Interval", "120 - 200 ms", 
         lambda v: f"{v * 1000:.0f} ms" if v < 1.0 else f"{v:.0f} ms", 
         lambda v: ("Elevated", "#f59e0b") if (v > 0.21 if v < 1.0 else v > 210) else (("Reduced", "#f59e0b") if (v < 0.12 if v < 1.0 else v < 120) else ("Normal", "#10b981"))),
        ("ecg_qtc_interval", "ECG / Cardiac", "QTc Interval", "350 - 450 ms", 
         lambda v: f"{v * 1000:.0f} ms" if v < 1.0 else f"{v:.0f} ms", 
         lambda v: ("Critical", "#ef4444") if (v > 0.48 if v < 1.0 else v > 480) else ("Normal", "#10b981")),
        ("ecg_qrs_amplitude", "ECG / Cardiac", "QRS Amplitude", "0.8 - 1.8 mV", 
         lambda v: f"{v:.2f} mV", 
         lambda v: ("Reduced", "#f59e0b") if v < 0.8 else (("Elevated", "#f59e0b") if v > 1.8 else ("Normal", "#10b981"))),
        ("ecg_p_amplitude", "ECG / Cardiac", "P Wave Amplitude", "0.05 - 0.20 mV", 
         lambda v: f"{v:.2f} mV", 
         lambda v: ("Normal", "#10b981")),
        ("ecg_t_amplitude", "ECG / Cardiac", "T Wave Amplitude", "0.15 - 0.40 mV", 
         lambda v: f"{v:.2f} mV", 
         lambda v: ("Reduced", "#f59e0b") if v < 0.10 else ("Normal", "#10b981")),
        ("hrv_sdnn", "ECG / Cardiac", "SDNN (HRV)", "30 - 100 ms", 
         lambda v: f"{v * 1000:.1f} ms" if v < 1.0 else f"{v:.1f} ms", 
         lambda v: ("Reduced", "#f59e0b") if (v < 0.02 if v < 1.0 else v < 20.0) else ("Normal", "#10b981")),
        ("hrv_rmssd", "ECG / Cardiac", "RMSSD (Vagal)", "20 - 80 ms", 
         lambda v: f"{v * 1000:.1f} ms" if v < 1.0 else f"{v:.1f} ms", 
         lambda v: ("Reduced", "#f59e0b") if (v < 0.015 if v < 1.0 else v < 15.0) else ("Normal", "#10b981")),
        ("hrv_pnn50", "ECG / Cardiac", "pNN50", "3 - 50 %", 
         lambda v: f"{v:.1f} %", 
         lambda v: ("Normal", "#10b981")),
        
        # PPG Features
        ("ppg_pulse_rate", "PPG / Perfusion", "PPG Pulse Rate", "60 - 100 bpm", 
         lambda v: f"{v:.1f} bpm", 
         lambda v: ("Critical", "#ef4444") if (v < 45 or v > 130) else (("Elevated", "#f59e0b") if v > 100 else (("Reduced", "#f59e0b") if v < 60 else ("Normal", "#10b981")))),
        ("ppg_ptt", "PPG / Perfusion", "Pulse Transit Time (PTT)", "180 - 320 ms", 
         lambda v: f"{v * 1000:.0f} ms" if v < 1.0 else f"{v:.0f} ms", 
         lambda v: ("Elevated", "#f59e0b") if (v > 0.35 if v < 1.0 else v > 350) else (("Reduced", "#f59e0b") if (v < 0.15 if v < 1.0 else v < 150) else ("Normal", "#10b981"))),
        ("ppg_pulse_amplitude", "PPG / Perfusion", "Pulse Amplitude", "0.3 - 0.9", 
         lambda v: f"{v:.3f}", 
         lambda v: ("Critical", "#ef4444") if v < 0.12 else (("Reduced", "#f59e0b") if v < 0.3 else ("Normal", "#10b981"))),
        ("ppg_pulse_width", "PPG / Perfusion", "Pulse Width", "150 - 280 ms", 
         lambda v: f"{v * 1000:.0f} ms" if v < 1.0 else f"{v:.0f} ms", 
         lambda v: ("Normal", "#10b981")),
        ("ppg_dicrotic_notch_pos", "PPG / Perfusion", "Notch Relative Position", "0.35 - 0.50", 
         lambda v: f"{v:.2f}", 
         lambda v: ("Normal", "#10b981")),
        ("ppg_reflection_index", "PPG / Perfusion", "Reflection Index", "25 - 45 %", 
         lambda v: f"{v * 100:.1f} %" if v < 1.0 else f"{v:.1f} %", 
         lambda v: ("Normal", "#10b981")),
        ("ppg_stiffness_index", "PPG / Perfusion", "Stiffness Index", "5.0 - 9.0 m/s", 
         lambda v: f"{v:.2f} m/s", 
         lambda v: ("Elevated", "#f59e0b") if v > 10.0 else ("Normal", "#10b981")),
        ("ppg_perfusion_index", "PPG / Perfusion", "Perfusion Index", "1.0 - 3.0 %", 
         lambda v: f"{v:.2f} %", 
         lambda v: ("Reduced", "#f59e0b") if v < 0.8 else ("Normal", "#10b981")),
        ("ppg_pulse_variability", "PPG / Perfusion", "Pulse Variability", "0.01 - 0.08", 
         lambda v: f"{v:.3f}", 
         lambda v: ("Elevated", "#f59e0b") if v > 0.12 else ("Normal", "#10b981")),
        
        # Respiratory Features
        ("rsp_rate", "Respiratory", "Respiratory Rate", "12 - 20 bpm", 
         lambda v: f"{v:.1f} bpm", 
         lambda v: ("Critical", "#ef4444") if (v < 8 or v > 32) else (("Elevated", "#f59e0b") if v > 20 else (("Reduced", "#f59e0b") if v < 12 else ("Normal", "#10b981")))),
        ("rsp_insp_duration", "Respiratory", "Inspiration Duration", "0.8 - 1.5 s", 
         lambda v: f"{v:.2f} s", 
         lambda v: ("Normal", "#10b981")),
        ("rsp_exp_duration", "Respiratory", "Expiration Duration", "1.2 - 2.5 s", 
         lambda v: f"{v:.2f} s", 
         lambda v: ("Normal", "#10b981")),
        ("rsp_ie_ratio", "Respiratory", "I:E Ratio", "1:1.20 - 1:2.00", 
         lambda v: f"1:{v:.2f}", 
         lambda v: ("Elevated", "#f59e0b") if (v > 2.2 or v < 1.0) else ("Normal", "#10b981")),
        ("rsp_variability", "Respiratory", "Breathing Variability", "0.05 - 0.20", 
         lambda v: f"{v:.3f}", 
         lambda v: ("Reduced", "#f59e0b") if v < 0.03 else ("Normal", "#10b981")),
        
        # Oxygenation Features
        ("spo2", "Oxygenation", "Oxygen Saturation (SpO₂)", "95 - 100 %", 
         lambda v: f"{v:.1f} %", 
         lambda v: ("Critical", "#ef4444") if v < 92 else (("Reduced", "#f59e0b") if v < 95 else ("Normal", "#10b981"))),
        ("spo2_desat_events", "Oxygenation", "Desaturation Incidents", "0 events", 
         lambda v: f"{v:.0f}", 
         lambda v: ("Critical", "#ef4444") if v >= 2.0 else (("Elevated", "#f59e0b") if v >= 1.0 else ("Normal", "#10b981"))),
        ("spo2_perfusion_quality", "Oxygenation", "Perfusion Quality Index", "0.90 - 1.00", 
         lambda v: f"{v:.3f}", 
         lambda v: ("Reduced", "#f59e0b") if v < 0.88 else ("Normal", "#10b981")),
    ]

    def __init__(self):
        super().__init__()
        self.setObjectName("FeaturesPage")
        
        # Patient info defaults
        import os
        self.patient_name = os.getenv("PATIENT_NAME", "John Doe")
        self.patient_ward = os.getenv("PATIENT_WARD", "ICU Ward A")
        self.patient_unit = "Unit 04"
        self.patient_bed = os.getenv("PATIENT_BED", "Bed 12")
        if not self.patient_bed or self.patient_bed.lower() == "none":
            self.patient_bed = "Bed 12"
            
        self._init_ui()

    def _init_ui(self):
        master = QVBoxLayout(self)
        master.setContentsMargins(0, 0, 0, 0)
        master.setSpacing(0)
        self.setStyleSheet("background-color: #020617;")

        # ── 1. HEADER ──────────────────────────────────────────────────
        header = QFrame()
        header.setObjectName("HFHeader")
        header.setFixedHeight(56)
        header.setStyleSheet("""
            QFrame#HFHeader {
                background-color: #0d1526;
                border-bottom: 1px solid #1e3352;
            }
        """)
        hdr_lay = QHBoxLayout(header)
        hdr_lay.setContentsMargins(20, 0, 20, 0)
        hdr_lay.setSpacing(0)

        brand = QLabel("HealthFi")
        brand.setFont(QFont("Inter", 15, QFont.Weight.Bold))
        brand.setStyleSheet("color: #22d3ee; letter-spacing: 1px; background: transparent;")
        hdr_lay.addWidget(brand)

        div1 = self._vdivider()
        hdr_lay.addWidget(div1)

        self.lbl_patient = QLabel()
        self.lbl_patient.setContentsMargins(12, 0, 12, 0)
        hdr_lay.addWidget(self.lbl_patient)

        self.lbl_location = QLabel()
        self.lbl_location.setContentsMargins(0, 0, 16, 0)
        hdr_lay.addWidget(self.lbl_location)

        conn = QLabel("● LIVE")
        conn.setFont(QFont("JetBrains Mono", 9, QFont.Weight.Bold))
        conn.setStyleSheet("color: #10b981; background: transparent;")
        hdr_lay.addWidget(conn)

        hdr_lay.addStretch()

        self.demo_timer_label = QLabel("⬤  Active Monitoring")
        self.demo_timer_label.setFont(QFont("JetBrains Mono", 8, QFont.Weight.Bold))
        self.demo_timer_label.setStyleSheet("""
            color: #10b981;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid #10b981;
            border-radius: 4px;
            padding: 4px 8px;
        """)
        hdr_lay.addWidget(self.demo_timer_label)
        master.addWidget(header)

        # ── 2. BODY ────────────────────────────────────────────────────
        body = QWidget()
        body_lay = QHBoxLayout(body)
        body_lay.setContentsMargins(0, 0, 0, 0)
        body_lay.setSpacing(0)

        # ── LEFT SIDEBAR ──────────────────────────────────────────────
        sidebar = QFrame()
        sidebar.setObjectName("HFSidebar")
        sidebar.setFixedWidth(188)
        sidebar.setStyleSheet("""
            QFrame#HFSidebar {
                background-color: #080f1e;
                border-right: 1px solid #1e2d45;
            }
        """)
        sb_lay = QVBoxLayout(sidebar)
        sb_lay.setContentsMargins(12, 18, 12, 18)
        sb_lay.setSpacing(6)

        self.lbl_ward = QLabel()
        self.lbl_ward.setFont(QFont("Inter", 11, QFont.Weight.Bold))
        self.lbl_ward.setStyleSheet("color: #e2e8f0; background: transparent;")
        sb_lay.addWidget(self.lbl_ward)

        self.lbl_unit = QLabel()
        self.lbl_unit.setFont(QFont("JetBrains Mono", 8))
        self.lbl_unit.setStyleSheet("color: #475569; margin-bottom: 12px; background: transparent;")
        sb_lay.addWidget(self.lbl_unit)

        # Sidebar menu buttons
        self.btn_live = self._sidebar_btn("📡  Live Monitor")
        self.btn_live.clicked.connect(self.view_live_requested.emit)
        self.btn_features = self._sidebar_btn("📊  Tabular Features", active=True)
        self.btn_reports = self._sidebar_btn("📋  Reports")
        self.btn_reports.clicked.connect(self.report_requested.emit)
        self.btn_history = self._sidebar_btn("🗂  History")
        self.btn_history.clicked.connect(self.view_logs_requested.emit)
        self.btn_setup = self._sidebar_btn("⚙️  Setup")
        self.btn_setup.clicked.connect(self.stop_requested.emit)

        for b in (self.btn_live, self.btn_features, self.btn_reports, self.btn_history, self.btn_setup):
            sb_lay.addWidget(b)

        sb_lay.addWidget(self._hdivider())

        # Temperature section in sidebar
        temp_title = QLabel("🌡️  Temperature")
        temp_title.setFont(QFont("Inter", 9, QFont.Weight.Bold))
        temp_title.setStyleSheet("color: #e2e8f0; background: transparent;")
        sb_lay.addWidget(temp_title)

        self.lbl_temp_core = QLabel("Core: 37.00 °C")
        self.lbl_temp_core.setFont(QFont("JetBrains Mono", 8, QFont.Weight.Bold))
        self.lbl_temp_core.setStyleSheet("color: #38bdf8; background: transparent;")
        sb_lay.addWidget(self.lbl_temp_core)

        self.lbl_temp_skin = QLabel("Skin: 35.50 °C")
        self.lbl_temp_skin.setFont(QFont("JetBrains Mono", 8, QFont.Weight.Bold))
        self.lbl_temp_skin.setStyleSheet("color: #a7f3d0; background: transparent;")
        sb_lay.addWidget(self.lbl_temp_skin)

        sb_lay.addWidget(self._hdivider())
        sb_lay.addStretch()

        body_lay.addWidget(sidebar)

        # ── 3. FEATURES CENTRAL DASHBOARD ─────────────────────────────
        dash = QWidget()
        dash_lay = QVBoxLayout(dash)
        dash_lay.setContentsMargins(20, 20, 20, 20)
        dash_lay.setSpacing(14)

        # Title block
        title_box = QHBoxLayout()
        title_lbl = QLabel("📊  Tabular Feature Engineering Layer")
        title_lbl.setFont(QFont("Inter", 14, QFont.Weight.Bold))
        title_lbl.setStyleSheet("color: #f1f5f9; background: transparent;")
        title_box.addWidget(title_lbl)
        
        self.lbl_score_badge = QLabel("Anomaly Score: 5.0 (Normal)")
        self.lbl_score_badge.setFont(QFont("JetBrains Mono", 9, QFont.Weight.Bold))
        self.lbl_score_badge.setStyleSheet("""
            color: #10b981;
            background-color: rgba(16, 185, 129, 0.1);
            border: 1px solid #10b981;
            border-radius: 6px;
            padding: 5px 12px;
        """)
        title_box.addStretch()
        title_box.addWidget(self.lbl_score_badge)
        dash_lay.addLayout(title_box)

        # SHAP Contributing features summary widget
        self.shap_box = QFrame()
        self.shap_box.setObjectName("SHAPBox")
        self.shap_box.setStyleSheet("""
            QFrame#SHAPBox {
                background-color: rgba(15, 23, 42, 0.8);
                border: 1px dashed #1e3352;
                border-radius: 8px;
            }
        """)
        self.shap_lay = QHBoxLayout(self.shap_box)
        self.shap_lay.setContentsMargins(14, 10, 14, 10)
        
        self.lbl_shap_title = QLabel("🤖  AI Clinical Drivers (SHAP):")
        self.lbl_shap_title.setFont(QFont("Inter", 8, QFont.Weight.Bold))
        self.lbl_shap_title.setStyleSheet("color: #94a3b8; background: transparent;")
        self.shap_lay.addWidget(self.lbl_shap_title)
        
        self.lbl_shap_cont1 = QLabel("NSR Baseline")
        self.lbl_shap_cont1.setFont(QFont("JetBrains Mono", 8, QFont.Weight.Bold))
        self.lbl_shap_cont1.setStyleSheet("color: #64748b; background: transparent;")
        self.shap_lay.addWidget(self.lbl_shap_cont1)
        
        self.shap_lay.addStretch()
        dash_lay.addWidget(self.shap_box)

        # Feature table scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setObjectName("FeatureScroll")
        scroll.setStyleSheet("""
            QScrollArea#FeatureScroll {
                background: transparent;
                border: none;
            }
        """)

        # The core table widget
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "System Category", "Physiological Feature Name", "Live Value", "Normal Baseline", "Drift Status", "SHAP Weight"
        ])
        
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #070d19;
                color: #e2e8f0;
                gridline-color: #1e293b;
                border: 1px solid #1e3352;
                border-radius: 8px;
            }
            QHeaderView::section {
                background-color: #0d1526;
                color: #94a3b8;
                padding: 6px;
                border: 1px solid #1e293b;
                font-family: "Inter";
                font-weight: bold;
                font-size: 10px;
            }
        """)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setRowCount(len(self.FEATURE_DEFS))
        
        self._populate_static_features()
        scroll.setWidget(self.table)
        dash_lay.addWidget(scroll)

        body_lay.addWidget(dash, stretch=1)
        master.addWidget(body, stretch=1)

        self.update_patient_ui()

    @staticmethod
    def _vdivider():
        d = QFrame()
        d.setFrameShape(QFrame.Shape.VLine)
        d.setFixedWidth(1)
        d.setStyleSheet("background: #1e3352; border: none; margin: 10px 12px;")
        return d

    @staticmethod
    def _hdivider():
        d = QFrame()
        d.setFrameShape(QFrame.Shape.HLine)
        d.setFixedHeight(1)
        d.setStyleSheet("background: #1e3352; border: none; margin: 6px 0px;")
        return d

    def _sidebar_btn(self, text, active=False):
        btn = QPushButton(text)
        btn.setFont(QFont("JetBrains Mono", 8, QFont.Weight.Bold))
        if active:
            btn.setStyleSheet("""
                QPushButton {
                    background: #0e3a5c;
                    color: #38bdf8;
                    border: 1px solid #0369a1;
                    border-radius: 6px;
                    padding: 9px;
                    text-align: left;
                }
            """)
        else:
            btn.setStyleSheet("""
                QPushButton {
                    background: transparent;
                    color: #64748b;
                    border: none;
                    border-radius: 6px;
                    padding: 9px;
                    text-align: left;
                }
                QPushButton:hover {
                    background: #0a1f3d;
                    color: #cbd5e1;
                }
            """)
        return btn

    def _populate_static_features(self):
        """Populate initial names, categories, and normal ranges of features."""
        for row, (_, category, label, range_str, _, _) in enumerate(self.FEATURE_DEFS):
            # Category
            cat_item = QTableWidgetItem(category)
            cat_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            cat_item.setForeground(QColor("#cbd5e1"))
            cat_item.setFont(QFont("Inter", 8, QFont.Weight.Bold))
            self.table.setItem(row, 0, cat_item)

            # Name
            name_item = QTableWidgetItem(label)
            name_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            name_item.setForeground(QColor("#94a3b8"))
            name_item.setFont(QFont("Inter", 8))
            self.table.setItem(row, 1, name_item)

            # Expected Range
            range_item = QTableWidgetItem(range_str)
            range_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            range_item.setForeground(QColor("#475569"))
            range_item.setFont(QFont("JetBrains Mono", 8))
            self.table.setItem(row, 3, range_item)

            # Initialize empty live value, status, and SHAP items
            val_item = QTableWidgetItem("—")
            val_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            val_item.setFont(QFont("JetBrains Mono", 8, QFont.Weight.Bold))
            val_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 2, val_item)

            status_item = QTableWidgetItem("Normal")
            status_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            status_item.setFont(QFont("Inter", 8, QFont.Weight.Bold))
            status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 4, status_item)

            shap_item = QTableWidgetItem("0.00")
            shap_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            shap_item.setFont(QFont("JetBrains Mono", 8))
            shap_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row, 5, shap_item)
            
            self.table.setRowHeight(row, 24)

        # Set column widths nicely
        self.table.setColumnWidth(0, 120)
        self.table.setColumnWidth(1, 240)
        self.table.setColumnWidth(2, 100)
        self.table.setColumnWidth(3, 140)
        self.table.setColumnWidth(4, 100)
        self.table.setColumnWidth(5, 120)

    # ──────────────────────────────────────────────────────────────────
    # Public API – called by app.py / worker signals
    # ──────────────────────────────────────────────────────────────────
    def update_vitals(self, vitals: dict):
        """Processes the latest vitals, updates feature table rows, severity score, and SHAP influences."""
        # 1. Update sidebar temperatures
        temp_core = vitals.get("core_temperature", 37.0)
        temp_skin = vitals.get("skin_temperature", 35.5)
        self.lbl_temp_core.setText(f"Core: {temp_core:.2f} °C")
        self.lbl_temp_skin.setText(f"Skin: {temp_skin:.2f} °C")

        # 2. Update score badge
        score = vitals.get("anomaly_score", 5.0)
        severity = vitals.get("anomaly_severity", "Normal")
        self.lbl_score_badge.setText(f"Anomaly Score: {score:.1f} ({severity.upper()})")
        
        # Color badge depending on severity
        if severity == "Normal":
            self.lbl_score_badge.setStyleSheet("""
                color: #10b981;
                background-color: rgba(16, 185, 129, 0.1);
                border: 1px solid #10b981;
                border-radius: 6px;
                padding: 5px 12px;
            """)
        elif severity == "Mild":
            self.lbl_score_badge.setStyleSheet("""
                color: #fb7185;
                background-color: rgba(251, 113, 133, 0.1);
                border: 1px solid #f43f5e;
                border-radius: 6px;
                padding: 5px 12px;
            """)
        elif severity == "Moderate":
            self.lbl_score_badge.setStyleSheet("""
                color: #fb923c;
                background-color: rgba(251, 146, 60, 0.1);
                border: 1px solid #ea580c;
                border-radius: 6px;
                padding: 5px 12px;
            """)
        else: # Severe
            self.lbl_score_badge.setStyleSheet("""
                color: #f43f5e;
                background-color: rgba(244, 63, 94, 0.2);
                border: 2px solid #e11d48;
                border-radius: 6px;
                padding: 5px 12px;
            """)

        # 3. Update SHAP banner
        shap_list = vitals.get("shap_contributors", [])
        if shap_list:
            text_items = []
            for item in shap_list:
                feature_name = item["feature"].replace("_", " ").upper()
                inf = item["influence"]
                text_items.append(f"<b style='color:#f43f5e;'>{feature_name}</b> ({inf:+.2f})")
            self.lbl_shap_cont1.setText("  |  ".join(text_items))
            self.lbl_shap_cont1.setStyleSheet("color: #e2e8f0; background: transparent;")
        else:
            self.lbl_shap_cont1.setText("Normal Baseline (No abnormal drivers)")
            self.lbl_shap_cont1.setStyleSheet("color: #64748b; background: transparent;")

        # Map SHAP features for quick lookup
        shap_map = {item["feature"]: item["influence"] for item in shap_list}

        # 4. Update Table Rows
        features = vitals.get("extracted_features", {})
        for row, (key, _, _, _, fmt_func, check_drift) in enumerate(self.FEATURE_DEFS):
            val = features.get(key)
            
            # Update Live Value item
            val_item = self.table.item(row, 2)
            if val_item:
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    val_item.setText(fmt_func(val))
                    val_item.setForeground(QColor("#38bdf8"))
                else:
                    val_item.setText("—")
                    val_item.setForeground(QColor("#475569"))

            # Update Drift Status item
            status_item = self.table.item(row, 4)
            if status_item:
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    st_text, st_color = check_drift(val)
                    status_item.setText(st_text)
                    status_item.setForeground(QColor(st_color))
                else:
                    status_item.setText("—")
                    status_item.setForeground(QColor("#475569"))

            # Update SHAP Weight item
            shap_item = self.table.item(row, 5)
            if shap_item:
                influence = shap_map.get(key, 0.0)
                if abs(influence) > 1e-4:
                    shap_item.setText(f"{influence:+.3f}")
                    # Color based on push direction
                    if influence > 0:
                        shap_item.setForeground(QColor("#f43f5e")) # pushes score up (red)
                    else:
                        shap_item.setForeground(QColor("#34d399")) # stable/reducing (green)
                else:
                    shap_item.setText("0.00")
                    shap_item.setForeground(QColor("#475569"))

    def update_demo_countdown(self, seconds_left):
        """Shows remaining demo time in header status badge."""
        mins, secs = divmod(seconds_left, 60)
        self.demo_timer_label.setText(f"⬤  Demo: {mins:02d}:{secs:02d}")
        self.demo_timer_label.setStyleSheet("""
            color: #22d3ee;
            background: rgba(6, 182, 212, 0.1);
            border: 1px solid #0891b2;
            border-radius: 4px;
            padding: 4px 8px;
        """)

    def set_active_monitoring_label(self):
        """Resets header badge to active monitoring state."""
        self.demo_timer_label.setText("⬤  Active Monitoring")
        self.demo_timer_label.setStyleSheet("""
            color: #10b981;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid #10b981;
            border-radius: 4px;
            padding: 4px 8px;
        """)

    def update_patient_ui(self):
        self.lbl_patient.setText(f"<span style='color:#475569;font-size:9px;'>PATIENT</span><br/><b style='color:#e2e8f0;font-size:11px;'>{self.patient_name}</b>")
        self.lbl_location.setText(f"<span style='color:#475569;font-size:9px;'>LOCATION</span><br/><b style='color:#e2e8f0;font-size:11px;'>{self.patient_unit} / {self.patient_bed}</b>")
        self.lbl_ward.setText(f"🏥  {self.patient_ward}")
        self.lbl_unit.setText(f"{self.patient_unit} — {self.patient_bed}")

    def update_patient_info(self, info):
        self.patient_name = info.get("name", "John Doe")
        self.patient_ward = info.get("ward", "ICU Ward A")
        self.patient_unit = info.get("unit", "Unit 04")
        self.patient_bed = info.get("bed", "Bed 12")
        self.update_patient_ui()

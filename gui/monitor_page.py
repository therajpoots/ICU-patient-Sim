"""
Bedside Patient Monitor View Page - Glassmorphic Premium Redesign.
Features:
  - High-fidelity QPainter-based SweepPlotter with sweep cursor dot and clinical grid
  - Signal display uses pre-filtered waveforms (clean ECG, clean PPG, clean RSP)
  - All 20 extracted features displayed in a scrollable panel
  - Glassmorphic card effects with glow borders
  - Flashing critical alarm banner
  - Modern sidebar with status indicators
"""

import time
import collections
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QFrame, QTableWidget, QTableWidgetItem, QHeaderView,
    QListWidget, QListWidgetItem, QScrollArea, QSizePolicy,
    QDialog, QGridLayout, QLineEdit, QMessageBox
)
from PyQt6.QtCore import pyqtSignal, Qt, QTimer
from PyQt6.QtGui import (
    QFont, QColor, QPainter, QPen, QPainterPath, QBrush
)


# ──────────────────────────────────────────────────────────────────────────────
# SweepPlotter – high-fidelity native waveform widget
# ──────────────────────────────────────────────────────────────────────────────
class SweepPlotter(QWidget):
    """
    Premium clinical sweep-display plotter using QPainter.
    - Rolling circular buffer of 750 samples (3s @ 250 Hz)
    - Adaptive pop-rate to match 250 Hz display speed
    - Fine grid, title label, and sweep-head cursor dot
    - Auto-scales signal between min_val and max_val with margin
    """

    def __init__(self, title, color_hex, min_val=-2.0, max_val=2.0, parent=None):
        super().__init__(parent)
        self.title = title
        self.color = QColor(color_hex)
        # Semi-transparent version for glow effect
        self.glow_color = QColor(
            self.color.red(), self.color.green(), self.color.blue(), 35
        )
        self.min_val = min_val
        self.max_val = max_val
        self.setMinimumHeight(90)

        # Rolling buffer: 2500 samples = 10 seconds at 250 Hz
        self.buffer_size = 2500
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.write_pos = 0

        # Incoming sample queue
        self.queue = collections.deque(maxlen=5000)

        # 25 Hz redraw timer (40 ms) — only active when widget is visible
        self.timer = QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self._tick)
        # Do NOT start here — start/stop via showEvent/hideEvent to avoid
        # crashes when QTimer fires during window-flag transitions

        self.setAutoFillBackground(False)

    def showEvent(self, event):
        """Start render timer only when widget becomes visible."""
        super().showEvent(event)
        if not self.timer.isActive():
            self.timer.start()

    def hideEvent(self, event):
        """Stop render timer when widget is hidden or during transitions."""
        super().hideEvent(event)
        self.timer.stop()

    def cleanup(self):
        """Explicitly stop timer — call before destroying the widget."""
        self.timer.stop()
        self.queue.clear()

    def add_samples(self, samples):
        """Queue new samples for display."""
        self.queue.extend(samples)

    def _tick(self):
        """Consume queued samples into ring buffer."""
        q_len = len(self.queue)
        if q_len == 0:
            return
        # Adaptive pop rate to maintain realtime display
        if q_len > 1500:
            pop_count = 25   # catch-up
        elif q_len > 750:
            pop_count = 15
        elif q_len < 80:
            pop_count = 7    # slow down slightly
        else:
            pop_count = 10   # nominal: 250 Hz / 25 Hz = 10
        pop_count = min(pop_count, q_len)
        for _ in range(pop_count):
            val = self.queue.popleft()
            self.buffer[self.write_pos] = val
            self.write_pos = (self.write_pos + 1) % self.buffer_size
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        if w < 10 or h < 10:
            return

        # ── Background: deep dark ──
        painter.fillRect(self.rect(), QColor("#070D1A"))

        # ── Subtle fine grid (5×5 every ~20px) ──
        grid_pen = QPen(QColor(255, 255, 255, 8), 1, Qt.PenStyle.SolidLine)
        painter.setPen(grid_pen)
        col_step = max(20, w // 20)
        row_step = max(20, h // 8)
        for x in range(0, w, col_step):
            painter.drawLine(x, 0, x, h)
        for y in range(0, h, row_step):
            painter.drawLine(0, y, w, y)

        # ── Major grid lines (brighter) every 5 minor ──
        major_pen = QPen(QColor(255, 255, 255, 20), 1, Qt.PenStyle.SolidLine)
        painter.setPen(major_pen)
        for x in range(0, w, col_step * 5):
            painter.drawLine(x, 0, x, h)
        for y in range(0, h, row_step * 4):
            painter.drawLine(0, y, w, y)

        # ── Zero-line ──
        zero_pen = QPen(QColor(255, 255, 255, 30), 1, Qt.PenStyle.DotLine)
        painter.setPen(zero_pen)
        zero_y = self._val_to_y(0.0, h)
        painter.drawLine(0, int(zero_y), w, int(zero_y))

        # ── Channel label top-left ──
        painter.setPen(self.color)
        font = QFont("JetBrains Mono", 8, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(10, 18, self.title)

        # ── Sweep gap: blanks gap_px pixels around write head ──
        GAP = 18   # pixels wide blank gap acting as sweep cursor

        # Current write head x position in screen coords
        head_x = int(self.write_pos * w / self.buffer_size)

        # Build two contiguous render segments that avoid the gap region
        # Segment A: old data from (write_pos + GAP_samples) to buffer_size
        gap_samples = max(1, int(GAP * self.buffer_size / w))
        seg_a_start = (self.write_pos + gap_samples) % self.buffer_size
        seg_a_end = self.buffer_size

        # Segment B: new data from 0 to (write_pos - gap_samples)
        seg_b_start = 0
        seg_b_end = max(0, (self.write_pos - gap_samples) % self.buffer_size)

        def draw_segment(start, end):
            if end <= start:
                return
            path = QPainterPath()
            first = True
            for idx in range(start, end):
                x = idx * w / self.buffer_size
                y = self._val_to_y(float(self.buffer[idx]), h)
                if first:
                    path.moveTo(x, y)
                    first = False
                else:
                    path.lineTo(x, y)
            if not first:
                # Single crisp line pass (no glow to avoid heap crash)
                line_pen = QPen(self.color, 1.5, Qt.PenStyle.SolidLine)
                line_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                line_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                painter.setPen(line_pen)
                painter.drawPath(path)

        draw_segment(seg_a_start, seg_a_end)
        draw_segment(seg_b_start, seg_b_end)

        # ── Sweep-head cursor: simple filled dot at write head ──
        safe_pos = max(0, self.write_pos - 1)
        dot_y = int(self._val_to_y(float(self.buffer[safe_pos]), h))
        # Outer glow ring
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(self.glow_color))
        painter.drawEllipse(head_x - 5, dot_y - 5, 10, 10)
        # Bright core dot
        painter.setBrush(QBrush(self.color))
        painter.drawEllipse(head_x - 3, dot_y - 3, 6, 6)
        # White center
        painter.setBrush(QBrush(QColor(255, 255, 255, 200)))
        painter.drawEllipse(head_x - 1, dot_y - 1, 3, 3)

    def _val_to_y(self, val: float, h: int) -> float:
        """Map signal value to screen y coordinate."""
        norm = (val - self.min_val) / max(1e-9, self.max_val - self.min_val)
        norm = max(0.0, min(1.0, norm))
        margin = 14
        return h - margin - norm * (h - 2 * margin)


# ──────────────────────────────────────────────────────────────────────────────
# Anomaly Feed Row
# ──────────────────────────────────────────────────────────────────────────────
class AnomalyFeedRowWidget(QWidget):
    """Compact styled event row for the live anomaly feed."""

    def __init__(self, timestamp, level, message, color_hex):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 4, 10, 4)
        layout.setSpacing(10)

        stripe = QFrame()
        stripe.setFixedSize(3, 28)
        stripe.setStyleSheet(f"background-color: {color_hex}; border-radius: 2px;")
        layout.addWidget(stripe)

        time_lbl = QLabel(timestamp)
        time_lbl.setFont(QFont("JetBrains Mono", 8))
        time_lbl.setStyleSheet("color: #64748b;")
        time_lbl.setFixedWidth(64)
        layout.addWidget(time_lbl)

        level_lbl = QLabel(level)
        level_lbl.setFont(QFont("JetBrains Mono", 7, QFont.Weight.Bold))
        level_lbl.setFixedWidth(52)
        level_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        level_lbl.setStyleSheet(f"""
            QLabel {{
                background-color: transparent;
                color: {color_hex};
                border: 1px solid {color_hex};
                border-radius: 3px;
                padding: 1px 3px;
            }}
        """)
        layout.addWidget(level_lbl)

        msg_lbl = QLabel(message)
        msg_lbl.setFont(QFont("Inter", 8))
        msg_lbl.setStyleSheet("color: #94a3b8;")
        msg_lbl.setWordWrap(False)
        layout.addWidget(msg_lbl, stretch=1)

        self.setStyleSheet("background: transparent;")


# ──────────────────────────────────────────────────────────────────────────────
# VitalCard – individual vital sign display
# ──────────────────────────────────────────────────────────────────────────────
class VitalCard(QFrame):
    """Glassmorphic vital sign card."""

    def __init__(self, label, name_key, init_val, unit, color_hex,
                 show_heart=False, show_mean_bp=False):
        super().__init__()
        self.name_key = name_key
        self.color = color_hex
        self.setObjectName(f"VitalCard_{name_key}")
        self.setStyleSheet(f"""
            QFrame#VitalCard_{name_key} {{
                background-color: rgba(15, 23, 42, 0.85);
                border: 1px solid {color_hex};
                border-radius: 10px;
            }}
        """)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 6, 12, 6)
        lay.setSpacing(2)

        # Header row
        hdr = QHBoxLayout()
        lbl_w = QLabel(label)
        lbl_w.setFont(QFont("JetBrains Mono", 7, QFont.Weight.Bold))
        lbl_w.setStyleSheet(f"color: {color_hex}; border: none; background: transparent;")
        unit_w = QLabel(unit)
        unit_w.setFont(QFont("JetBrains Mono", 7))
        unit_w.setStyleSheet("color: #475569; border: none; background: transparent;")
        hdr.addWidget(lbl_w)
        hdr.addStretch()
        hdr.addWidget(unit_w)
        lay.addLayout(hdr)

        # Value
        val_row = QHBoxLayout()
        if show_heart:
            self.heart_icon = QLabel("♥")
            self.heart_icon.setFont(QFont("Inter", 12))
            self.heart_icon.setStyleSheet(f"color: {color_hex}; border: none; background: transparent;")
            val_row.addWidget(self.heart_icon)

        self.val_label = QLabel(init_val)
        val_font_size = 20 if len(init_val) > 4 else 24
        self.val_label.setFont(QFont("Inter", val_font_size, QFont.Weight.Bold))
        self.val_label.setStyleSheet(f"color: {color_hex}; border: none; background: transparent;")
        self.val_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        val_row.addWidget(self.val_label, stretch=1)
        lay.addLayout(val_row)

        if show_mean_bp:
            self.mean_label = QLabel("MAP: 93")
            self.mean_label.setFont(QFont("JetBrains Mono", 7))
            self.mean_label.setStyleSheet("color: #475569; border: none; background: transparent;")
            self.mean_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            lay.addWidget(self.mean_label)
        else:
            self.mean_label = None

    def setValue(self, value_str):
        """Sets the vital value text and dynamically adjusts the font size to prevent clipping."""
        self.val_label.setText(value_str)
        val_font_size = 20 if len(value_str) > 4 else 24
        self.val_label.setFont(QFont("Inter", val_font_size, QFont.Weight.Bold))

    def checkValue(self, val, min_val, max_val):
        """Checks if value exceeds thresholds, highlighting background and border in flashing neon-red."""
        is_alert = False
        try:
            val_float = float(val)
            if min_val is not None and val_float < min_val:
                is_alert = True
            if max_val is not None and val_float > max_val:
                is_alert = True
        except ValueError:
            pass

        if is_alert:
            self.setStyleSheet(f"""
                QFrame#VitalCard_{self.name_key} {{
                    background-color: rgba(127, 29, 29, 0.6);
                    border: 2px solid #ef4444;
                    border-radius: 10px;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QFrame#VitalCard_{self.name_key} {{
                    background-color: rgba(15, 23, 42, 0.85);
                    border: 1px solid {self.color};
                    border-radius: 10px;
                }}
            """)


# ──────────────────────────────────────────────────────────────────────────────
# SetupDialog – Alarm Thresholds Dialog
# ──────────────────────────────────────────────────────────────────────────────
class SetupDialog(QDialog):
    """
    Clinician Setup Dialog for adjusting vital sign alert thresholds and ML sensitivity.
    """
    def __init__(self, current_settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Clinical Alarm Thresholds Setup")
        self.setObjectName("SetupDialog")
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        
        self.current_settings = current_settings
        self.stop_requested = False
        self.init_ui()

    def init_ui(self):
        # Dialog-wide dark-neon stylesheet matching design guidelines
        self.setStyleSheet("""
            QDialog#SetupDialog {
                background-color: transparent;
            }
            QFrame#SetupCard {
                background-color: #0c1524;
                border: 1px solid rgba(34, 211, 238, 0.35);
                border-radius: 16px;
            }
            QLabel#Title {
                color: #ffffff;
                font-family: 'Inter', sans-serif;
                font-size: 18px;
                font-weight: 800;
            }
            QLabel#Label {
                color: #94a3b8;
                font-family: 'JetBrains Mono', sans-serif;
                font-size: 10px;
                font-weight: bold;
                text-transform: uppercase;
            }
            QLineEdit {
                background-color: #05080f;
                border: 1px solid #1e2d45;
                border-radius: 6px;
                padding: 6px 10px;
                color: #f8fafc;
                font-family: 'JetBrains Mono';
                font-size: 12px;
            }
            QLineEdit:focus {
                border: 1px solid #22d3ee;
            }
            QPushButton {
                font-family: 'JetBrains Mono';
                font-size: 11px;
                font-weight: bold;
                border-radius: 6px;
                padding: 10px 16px;
            }
            QPushButton#SaveBtn {
                background-color: #06b6d4;
                color: #ffffff;
                border: none;
            }
            QPushButton#SaveBtn:hover {
                background-color: #22d3ee;
            }
            QPushButton#CancelBtn {
                background-color: transparent;
                color: #cbd5e1;
                border: 1px solid #1e2d45;
            }
            QPushButton#CancelBtn:hover {
                background-color: rgba(255, 255, 255, 0.05);
            }
            QPushButton#StopMonBtn {
                background-color: #7f1d1d;
                color: #fca5a5;
                border: 1px solid #ef4444;
            }
            QPushButton#StopMonBtn:hover {
                background-color: #ef4444;
                color: #ffffff;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        card = QFrame()
        card.setObjectName("SetupCard")
        card_lay = QVBoxLayout(card)
        card_lay.setContentsMargins(24, 24, 24, 24)
        card_lay.setSpacing(16)
        
        # Header
        hdr = QHBoxLayout()
        title = QLabel("⚙️  Clinical Monitor Setup")
        title.setObjectName("Title")
        hdr.addWidget(title)
        hdr.addStretch()
        
        close_btn = QPushButton("×")
        close_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #c6c6cd;
                font-size: 20px;
                font-weight: bold;
                border: none;
                padding: 0;
                min-width: 24px;
                max-width: 24px;
            }
            QPushButton:hover {
                color: #ffb4ab;
            }
        """)
        close_btn.clicked.connect(self.reject)
        hdr.addWidget(close_btn)
        card_lay.addLayout(hdr)
        
        # Grid of inputs
        grid = QGridLayout()
        grid.setSpacing(12)
        
        self.inputs = {}
        
        # Left side: Physiological Thresholds
        threshold_fields = [
            ("hr_min", "HR Min (bpm)", str(self.current_settings["hr_min"])),
            ("hr_max", "HR Max (bpm)", str(self.current_settings["hr_max"])),
            ("sbp_min", "SBP Min (mmHg)", str(self.current_settings["sbp_min"])),
            ("sbp_max", "SBP Max (mmHg)", str(self.current_settings["sbp_max"])),
            ("spo2_min", "SpO2 Min (%)", str(self.current_settings["spo2_min"])),
            ("rr_min", "RESP Min (rpm)", str(self.current_settings["rr_min"])),
            ("rr_max", "RESP Max (rpm)", str(self.current_settings["rr_max"])),
            ("ml_threshold", "ML Threshold (0-100)", str(self.current_settings["ml_threshold"])),
        ]
        
        row = 0
        for key, label_text, init_val in threshold_fields:
            lbl = QLabel(label_text)
            lbl.setObjectName("Label")
            edit = QLineEdit(init_val)
            edit.setFixedWidth(100)
            self.inputs[key] = edit
            grid.addWidget(lbl, row, 0)
            grid.addWidget(edit, row, 1)
            row += 1
            
        # Right side: Patient Demographics
        patient_fields = [
            ("patient_name", "Patient Name", self.current_settings["patient_name"]),
            ("patient_id", "Patient ID", self.current_settings["patient_id"]),
            ("patient_ward", "Ward", self.current_settings["patient_ward"]),
            ("patient_unit", "Unit", self.current_settings["patient_unit"]),
            ("patient_bed", "Bed / Location", self.current_settings["patient_bed"]),
            ("patient_age", "Age", self.current_settings["patient_age"]),
        ]
        
        row = 0
        for key, label_text, init_val in patient_fields:
            lbl = QLabel(label_text)
            lbl.setObjectName("Label")
            edit = QLineEdit(init_val)
            edit.setFixedWidth(140)
            self.inputs[key] = edit
            grid.addWidget(lbl, row, 2)
            grid.addWidget(edit, row, 3)
            row += 1
            
        card_lay.addLayout(grid)
        
        # Action Buttons
        btn_box = QHBoxLayout()
        btn_box.setSpacing(10)
        
        self.save_btn = QPushButton("Save Settings")
        self.save_btn.setObjectName("SaveBtn")
        self.save_btn.clicked.connect(self.on_save)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("CancelBtn")
        self.cancel_btn.clicked.connect(self.reject)
        
        self.stop_btn = QPushButton("Stop Monitoring")
        self.stop_btn.setObjectName("StopMonBtn")
        self.stop_btn.setToolTip("Return to Main Connection Settings Setup screen")
        self.stop_btn.clicked.connect(self.on_stop_monitoring)
        
        btn_box.addWidget(self.stop_btn)
        btn_box.addStretch()
        btn_box.addWidget(self.cancel_btn)
        btn_box.addWidget(self.save_btn)
        
        card_lay.addLayout(btn_box)
        layout.addWidget(card)
        self.setFixedSize(580, 420)

    def on_save(self):
        try:
            hr_min = float(self.inputs["hr_min"].text())
            hr_max = float(self.inputs["hr_max"].text())
            sbp_min = float(self.inputs["sbp_min"].text())
            sbp_max = float(self.inputs["sbp_max"].text())
            spo2_min = float(self.inputs["spo2_min"].text())
            rr_min = float(self.inputs["rr_min"].text())
            rr_max = float(self.inputs["rr_max"].text())
            ml_threshold = float(self.inputs["ml_threshold"].text())
            
            patient_name = self.inputs["patient_name"].text().strip()
            patient_id = self.inputs["patient_id"].text().strip()
            patient_ward = self.inputs["patient_ward"].text().strip()
            patient_unit = self.inputs["patient_unit"].text().strip()
            patient_bed = self.inputs["patient_bed"].text().strip()
            patient_age_str = self.inputs["patient_age"].text().strip()
            
            if hr_min >= hr_max:
                raise ValueError("Heart Rate Min must be less than Max.")
            if sbp_min >= sbp_max:
                raise ValueError("Systolic BP Min must be less than Max.")
            if rr_min >= rr_max:
                raise ValueError("Respiratory Rate Min must be less than Max.")
            if not (0 <= ml_threshold <= 100):
                raise ValueError("ML Threshold must be between 0 and 100.")
            if not (0 <= spo2_min <= 100):
                raise ValueError("SpO2 Min must be between 0 and 100.")
                
            if not patient_name:
                raise ValueError("Patient Name cannot be empty.")
            if not patient_id:
                raise ValueError("Patient ID cannot be empty.")
            if not patient_ward:
                raise ValueError("Ward cannot be empty.")
            if not patient_unit:
                raise ValueError("Unit cannot be empty.")
            if not patient_bed:
                raise ValueError("Bed cannot be empty.")
            try:
                patient_age = int(patient_age_str)
                if patient_age <= 0:
                    raise ValueError
            except ValueError:
                raise ValueError("Age must be a positive integer.")
                
            self.new_settings = {
                "hr_min": hr_min,
                "hr_max": hr_max,
                "sbp_min": sbp_min,
                "sbp_max": sbp_max,
                "spo2_min": spo2_min,
                "rr_min": rr_min,
                "rr_max": rr_max,
                "ml_threshold": ml_threshold,
                "patient_name": patient_name,
                "patient_id": patient_id,
                "patient_ward": patient_ward,
                "patient_unit": patient_unit,
                "patient_bed": patient_bed,
                "patient_age": str(patient_age)
            }
            self.accept()
        except ValueError as e:
            QMessageBox.critical(self, "Validation Error", f"Please check setup entries:\n{e}")

    def on_stop_monitoring(self):
        reply = QMessageBox.question(
            self,
            "STOP MONITORING",
            "Are you sure you want to stop clinical monitoring and return to connection setup?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.stop_requested = True
            self.accept()

    def get_settings(self):
        return self.new_settings
        
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton and hasattr(self, "_drag_position"):
            self.move(event.globalPosition().toPoint() - self._drag_position)
            event.accept()

# ──────────────────────────────────────────────────────────────────────────────
# MonitorPage – main monitor view
# ──────────────────────────────────────────────────────────────────────────────
class MonitorPage(QWidget):
    stop_requested = pyqtSignal()
    inject_anomaly_requested = pyqtSignal(str)
    report_requested = pyqtSignal()
    view_logs_requested = pyqtSignal()
    view_features_requested = pyqtSignal()
    emergency_triggered = pyqtSignal()
    patient_info_changed = pyqtSignal(dict)

    # All 20 features with display labels and format strings
    FEATURE_DEFS = [
        ("ecg_qrs_duration",  "QRS Duration",        "{:.0f} ms",   lambda v: v * 1000 if v < 2.0 else v),
        ("ecg_st_elevation",  "ST Elevation",         "{:.3f} mV",   None),
        ("hrv_sdnn",          "SDNN (HRV)",           "{:.1f} ms",   lambda v: v * 1000 if v < 2.0 else v),
        ("hrv_rmssd",         "RMSSD (Vagal)",        "{:.1f} ms",   lambda v: v * 1000 if v < 2.0 else v),
        ("hrv_pnn50",         "pNN50",                "{:.1f} %",    None),
        ("rsp_rate",          "Breathing Rate",       "{:.1f} bpm",  None),
        ("rsp_tidal_volume",  "Tidal Volume",         "{:.4f}",      None),
        ("rsp_ie_ratio",      "I:E Ratio",            "1:{:.2f}",    None),
        ("sbp_mean",          "SBP Mean",             "{:.1f} mmHg", None),
        ("sbp_var",           "SBP Variance",         "{:.2f}",      None),
        ("sbp_slope",         "SBP Slope",            "{:.3f}",      None),
        ("dbp_mean",          "DBP Mean",             "{:.1f} mmHg", None),
        ("dbp_var",           "DBP Variance",         "{:.2f}",      None),
        ("dbp_slope",         "DBP Slope",            "{:.3f}",      None),
        ("spo2_mean",         "SpO2 Mean",            "{:.1f} %",    None),
        ("spo2_var",          "SpO2 Variance",        "{:.4f}",      None),
        ("spo2_slope",        "SpO2 Slope",           "{:.4f}",      None),
        ("hr_mean",           "HR Mean",              "{:.1f} bpm",  None),
        ("hr_var",            "HR Variance",          "{:.2f}",      None),
        ("hr_slope",          "HR Slope",             "{:.3f}",      None),
    ]

    def __init__(self):
        super().__init__()
        self.setObjectName("MonitorPage")
        self.anomaly_active = False
        self.flash_state = False

        # Alert thresholds
        self.thresh_hr_min = 50
        self.thresh_hr_max = 120
        self.thresh_sbp_min = 90
        self.thresh_sbp_max = 140
        self.thresh_spo2_min = 90
        self.thresh_rr_min = 8
        self.thresh_rr_max = 25
        self.thresh_ml_threshold = 45.0
        
        # Emergency call banner state
        self.emergency_active_banner = False

        # Patient info
        import os
        self.patient_name = os.getenv("PATIENT_NAME", "John Doe")
        self.patient_id = os.getenv("PATIENT_ID", "PT-2026-9041")
        self.patient_ward = os.getenv("PATIENT_WARD", "ICU Ward A")
        self.patient_unit = "Unit 04"
        self.patient_bed = os.getenv("PATIENT_BED", "Bed 12")
        if not self.patient_bed or self.patient_bed.lower() == "none":
            self.patient_bed = "Bed 12"
        self.patient_age = os.getenv("PATIENT_AGE", "65")

        self.flash_timer = QTimer(self)
        self.flash_timer.setInterval(400)
        self.flash_timer.timeout.connect(self._toggle_alarm_flash)

        self._init_ui()

    # ──────────────────────────────────────────────────────────────────────
    # UI Construction
    # ──────────────────────────────────────────────────────────────────────
    def _init_ui(self):
        master = QVBoxLayout(self)
        master.setContentsMargins(0, 0, 0, 0)
        master.setSpacing(0)
        self.setStyleSheet("background-color: #020617;")

        # ── 1. TOP HEADER ──────────────────────────────────────────────
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

        # Brand
        brand = QLabel("HealthFi")
        brand.setFont(QFont("Inter", 15, QFont.Weight.Bold))
        brand.setStyleSheet("color: #22d3ee; letter-spacing: 1px; background: transparent;")
        hdr_lay.addWidget(brand)

        div1 = self._vdivider()
        hdr_lay.addWidget(div1)

        # Patient info
        self.lbl_patient = QLabel()
        self.lbl_patient.setContentsMargins(12, 0, 12, 0)
        hdr_lay.addWidget(self.lbl_patient)

        self.lbl_location = QLabel()
        self.lbl_location.setContentsMargins(0, 0, 16, 0)
        hdr_lay.addWidget(self.lbl_location)

        # Connected badge
        conn = QLabel("● LIVE")
        conn.setFont(QFont("JetBrains Mono", 9, QFont.Weight.Bold))
        conn.setStyleSheet("color: #10b981; background: transparent;")
        hdr_lay.addWidget(conn)

        hdr_lay.addStretch()

        # Status / mode label
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
        self.btn_live = self._sidebar_btn("📡  Live Monitor", active=True)
        self.btn_features = self._sidebar_btn("📊  Tabular Features")
        self.btn_features.clicked.connect(self.view_features_requested.emit)
        self.btn_reports = self._sidebar_btn("📋  Reports")
        self.btn_reports.clicked.connect(self.report_requested.emit)
        self.btn_history = self._sidebar_btn("🗂  History")
        self.btn_history.clicked.connect(self.view_logs_requested.emit)
        self.btn_setup = self._sidebar_btn("⚙️  Setup")
        self.btn_setup.clicked.connect(self._on_setup_clicked)

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

        # Emergency button
        self.btn_emerg = QPushButton("⚠  Emergency Call")
        self.btn_emerg.setFont(QFont("JetBrains Mono", 8, QFont.Weight.Bold))
        self.btn_emerg.setStyleSheet("""
            QPushButton {
                background-color: #7f1d1d;
                color: #fca5a5;
                border: 1px solid #ef4444;
                border-radius: 7px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #ef4444;
                color: white;
            }
        """)
        self.btn_emerg.clicked.connect(self._on_emergency_clicked)
        sb_lay.addWidget(self.btn_emerg)

        body_lay.addWidget(sidebar)

        # ── DASHBOARD AREA ────────────────────────────────────────────
        dash = QWidget()
        dash.setStyleSheet("background: #020617;")
        dash_lay = QVBoxLayout(dash)
        dash_lay.setContentsMargins(14, 14, 14, 14)
        dash_lay.setSpacing(10)

        # Control bar
        ctrl_bar = QFrame()
        ctrl_bar.setObjectName("CtrlBar")
        ctrl_bar.setFixedHeight(50)
        ctrl_bar.setStyleSheet("""
            QFrame#CtrlBar {
                background-color: #0a1628;
                border: 1px solid #1e3352;
                border-radius: 8px;
            }
        """)
        ctrl_lay = QHBoxLayout(ctrl_bar)
        ctrl_lay.setContentsMargins(14, 0, 14, 0)
        ctrl_lay.setSpacing(10)

        self.start_btn = QPushButton("▶  Start Monitoring")
        self.start_btn.setEnabled(False)
        self.start_btn.setFont(QFont("JetBrains Mono", 8, QFont.Weight.Bold))
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #10b981;
                color: #022c22;
                border: none;
                border-radius: 5px;
                padding: 7px 14px;
            }
            QPushButton:disabled {
                background-color: #1e2d45;
                color: #334155;
            }
        """)

        self.stop_btn = QPushButton("⏹  Stop")
        self.stop_btn.setEnabled(True)
        self.stop_btn.clicked.connect(self.stop_requested.emit)
        self.stop_btn.setFont(QFont("JetBrains Mono", 8, QFont.Weight.Bold))
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #94a3b8;
                border: 1px solid #334155;
                border-radius: 5px;
                padding: 7px 14px;
            }
            QPushButton:hover {
                background-color: #1e3352;
                color: #e2e8f0;
            }
        """)

        ctrl_lay.addWidget(self.start_btn)
        ctrl_lay.addWidget(self.stop_btn)
        ctrl_lay.addStretch()

        self.report_btn = QPushButton("📄  Generate Report")
        self.report_btn.clicked.connect(self.report_requested.emit)
        self.report_btn.setFont(QFont("JetBrains Mono", 8, QFont.Weight.Bold))
        self.report_btn.setStyleSheet("""
            QPushButton {
                background-color: #1e3352;
                color: #93c5fd;
                border: 1px solid #2563eb;
                border-radius: 5px;
                padding: 7px 14px;
            }
            QPushButton:hover {
                background-color: #2563eb;
                color: white;
            }
        """)
        ctrl_lay.addWidget(self.report_btn)
        dash_lay.addWidget(ctrl_bar)

        # Alarm banner
        self.alarm_banner = QLabel("✓  SYSTEM OPERATIONAL — PATIENT TELEMETRY ACTIVE")
        self.alarm_banner.setObjectName("AlarmBanner")
        self.alarm_banner.setFont(QFont("JetBrains Mono", 9, QFont.Weight.Bold))
        self.alarm_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.alarm_banner.setFixedHeight(32)
        self.alarm_banner.setStyleSheet(
            "background-color: rgba(5, 46, 22, 0.8); color: #4ade80;"
            "border: 1px solid #166534; border-radius: 5px;"
        )
        dash_lay.addWidget(self.alarm_banner)

        # Central split: waveforms left, vitals+features right
        center = QHBoxLayout()
        center.setSpacing(10)

        # ── Waveform Panel ────────────────────────────────────────────
        wave_frame = QFrame()
        wave_frame.setObjectName("WaveFrame")
        wave_frame.setStyleSheet("""
            QFrame#WaveFrame {
                background-color: #070D1A;
                border: 1px solid #1e3352;
                border-radius: 10px;
            }
        """)
        wave_lay = QVBoxLayout(wave_frame)
        wave_lay.setContentsMargins(8, 8, 8, 8)
        wave_lay.setSpacing(6)

        # Channel header row
        ch_hdr = QHBoxLayout()
        for ch_name, ch_color in [("ECG Lead II", "#22c55e"), ("SpO₂ Pleth", "#06b6d4"), ("RESP", "#f97316")]:
            dot = QLabel(f"● {ch_name}")
            dot.setFont(QFont("JetBrains Mono", 8, QFont.Weight.Bold))
            dot.setStyleSheet(f"color: {ch_color}; background: transparent;")
            ch_hdr.addWidget(dot)
            ch_hdr.addStretch()
        wave_lay.addLayout(ch_hdr)

        # ECG: typical range after cleaning is approx -0.3 to 1.5 mV
        self.ecg_plotter = SweepPlotter("ECG II",   "#22c55e", min_val=-0.4, max_val=1.8, parent=self)
        # PPG after cleaning: ~ -0.5 to 1.5
        self.ppg_plotter = SweepPlotter("SpO₂ Pleth", "#06b6d4", min_val=-0.6, max_val=1.6, parent=self)
        # RSP after cleaning: ~ -0.6 to 0.6
        self.rsp_plotter = SweepPlotter("RESP",     "#f97316", min_val=-0.8, max_val=0.8, parent=self)

        wave_lay.addWidget(self.ecg_plotter, stretch=3)

        sep1 = QFrame()
        sep1.setFixedHeight(1)
        sep1.setStyleSheet("background: #1e2d45;")
        wave_lay.addWidget(sep1)

        wave_lay.addWidget(self.ppg_plotter, stretch=2)

        sep2 = QFrame()
        sep2.setFixedHeight(1)
        sep2.setStyleSheet("background: #1e2d45;")
        wave_lay.addWidget(sep2)

        wave_lay.addWidget(self.rsp_plotter, stretch=2)

        center.addWidget(wave_frame, stretch=9)

        # ── Right panel: Vitals + Features + Inject ───────────────────
        right_panel = QWidget()
        right_panel.setStyleSheet("background: transparent;")
        right_lay = QVBoxLayout(right_panel)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(8)

        # Vital cards
        self.hr_card = VitalCard("HR / PR", "HR", "72", "bpm", "#22c55e", show_heart=True)
        self.spo2_card = VitalCard("SpO₂", "SpO2", "98", "%", "#06b6d4")
        self.rr_card = VitalCard("RESP", "RR", "16", "rpm", "#f97316")
        self.bp_card = VitalCard("NIBP", "BP", "120/80", "mmHg", "#e2e8f0", show_mean_bp=True)

        for card in (self.hr_card, self.spo2_card, self.rr_card, self.bp_card):
            right_lay.addWidget(card)

        # Features panel with scroll area
        feat_frame = QFrame()
        feat_frame.setObjectName("FeatFrame")
        feat_frame.setStyleSheet("""
            QFrame#FeatFrame {
                background-color: rgba(10, 22, 40, 0.9);
                border: 1px solid #1e3352;
                border-radius: 10px;
            }
        """)
        feat_frame_lay = QVBoxLayout(feat_frame)
        feat_frame_lay.setContentsMargins(8, 6, 8, 6)
        feat_frame_lay.setSpacing(4)

        feat_title = QLabel("⚙  SIGNAL FEATURES")
        feat_title.setFont(QFont("JetBrains Mono", 7, QFont.Weight.Bold))
        feat_title.setStyleSheet("color: #64748b; background: transparent; border: none;")
        feat_frame_lay.addWidget(feat_title)

        # Scrollable feature table
        self.feature_table = QTableWidget()
        self.feature_table.setColumnCount(2)
        self.feature_table.setRowCount(len(self.FEATURE_DEFS))
        self.feature_table.setHorizontalHeaderLabels(["Feature", "Value"])
        self.feature_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.feature_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.feature_table.horizontalHeader().resizeSection(1, 70)
        self.feature_table.verticalHeader().setVisible(False)
        self.feature_table.setShowGrid(False)
        self.feature_table.setAlternatingRowColors(False)
        self.feature_table.setObjectName("FeatTable")
        self.feature_table.setFont(QFont("JetBrains Mono", 7))
        self.feature_table.setFixedHeight(130)
        self.feature_table.setStyleSheet("""
            QTableWidget#FeatTable {
                background-color: transparent;
                border: none;
                color: #94a3b8;
                selection-background-color: #1e3352;
            }
            QHeaderView::section {
                background-color: #0a1628;
                color: #475569;
                padding: 3px;
                border: none;
                font-size: 7px;
            }
            QTableWidget::item {
                padding: 2px 4px;
                border-bottom: 1px solid #0f1f35;
            }
        """)
        self._populate_feature_names()
        feat_frame_lay.addWidget(self.feature_table)
        right_lay.addWidget(feat_frame, stretch=1)

        # Inject anomaly controls
        inject_frame = QFrame()
        inject_frame.setObjectName("InjectFrame")
        inject_frame.setStyleSheet("""
            QFrame#InjectFrame {
                background-color: rgba(127, 29, 29, 0.2);
                border: 1px solid #7f1d1d;
                border-radius: 8px;
            }
        """)
        inject_lay = QHBoxLayout(inject_frame)
        inject_lay.setContentsMargins(8, 6, 8, 6)
        inject_lay.setSpacing(6)

        self.anomaly_combo = QComboBox()
        self.anomaly_combo.addItems([
            "pvc", "sinus_tachycardia", "sinus_bradycardia",
            "atrial_fibrillation", "ventricular_fibrillation",
            "hypertensive_spike", "spo2_desaturation", "stable"
        ])
        self.anomaly_combo.setObjectName("AnomalyCombo")
        self.anomaly_combo.setFont(QFont("JetBrains Mono", 7))
        self.anomaly_combo.setStyleSheet("""
            QComboBox {
                background: #0a1628; color: #94a3b8;
                border: 1px solid #334155; border-radius: 4px;
                padding: 4px 6px; font-size: 7px;
            }
        """)

        self.inject_btn = QPushButton("⚡ Inject")
        self.inject_btn.setFont(QFont("JetBrains Mono", 8, QFont.Weight.Bold))
        self.inject_btn.clicked.connect(self._on_inject_clicked)
        self.inject_btn.setStyleSheet("""
            QPushButton {
                background: #7f1d1d; color: #fca5a5;
                border: 1px solid #ef4444; border-radius: 5px;
                padding: 5px 10px;
            }
            QPushButton:hover { background: #ef4444; color: white; }
        """)
        inject_lay.addWidget(self.anomaly_combo, stretch=1)
        inject_lay.addWidget(self.inject_btn)
        right_lay.addWidget(inject_frame)

        center.addWidget(right_panel, stretch=3)
        dash_lay.addLayout(center, stretch=1)

        # ── Anomaly Feed ──────────────────────────────────────────────
        feed_frame = QFrame()
        feed_frame.setObjectName("FeedFrame")
        feed_frame.setFixedHeight(120)
        feed_frame.setStyleSheet("""
            QFrame#FeedFrame {
                background-color: #070d1a;
                border: 1px solid #1e2d45;
                border-radius: 8px;
            }
        """)
        feed_lay = QVBoxLayout(feed_frame)
        feed_lay.setContentsMargins(0, 0, 0, 0)
        feed_lay.setSpacing(0)

        feed_hdr = QFrame()
        feed_hdr.setFixedHeight(30)
        feed_hdr.setStyleSheet(
            "background: #0a1628; border-bottom: 1px solid #1e2d45;"
            "border-top-left-radius: 8px; border-top-right-radius: 8px;"
        )
        fhdr_lay = QHBoxLayout(feed_hdr)
        fhdr_lay.setContentsMargins(12, 0, 12, 0)
        feed_title = QLabel("⚠  ANOMALY EVENT LOG")
        feed_title.setFont(QFont("JetBrains Mono", 7, QFont.Weight.Bold))
        feed_title.setStyleSheet("color: #64748b; background: transparent;")
        fhdr_lay.addWidget(feed_title)
        feed_lay.addWidget(feed_hdr)

        self.feed_list = QListWidget()
        self.feed_list.setStyleSheet("""
            QListWidget {
                background: transparent;
                border: none;
                border-bottom-left-radius: 8px;
                border-bottom-right-radius: 8px;
            }
            QListWidget::item:hover {
                background: rgba(30, 51, 82, 0.3);
            }
        """)
        feed_lay.addWidget(self.feed_list)
        dash_lay.addWidget(feed_frame)

        body_lay.addWidget(dash)
        master.addWidget(body, stretch=1)

        self.update_patient_ui()

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────
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

    def _populate_feature_names(self):
        """Fill feature name column in the table."""
        for row, (_, label, _, _) in enumerate(self.FEATURE_DEFS):
            name_item = QTableWidgetItem(label)
            name_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            name_item.setForeground(QColor("#475569"))
            self.feature_table.setItem(row, 0, name_item)
            name_item.setFont(QFont("JetBrains Mono", 7))

            val_item = QTableWidgetItem("—")
            val_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            val_item.setForeground(QColor("#94a3b8"))
            val_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            val_item.setFont(QFont("JetBrains Mono", 7))
            self.feature_table.setItem(row, 1, val_item)
            self.feature_table.setRowHeight(row, 18)

    def _on_inject_clicked(self):
        self.inject_anomaly_requested.emit(self.anomaly_combo.currentText())

    # ──────────────────────────────────────────────────────────────────
    # Public API – called by app.py / worker signals
    # ──────────────────────────────────────────────────────────────────
    def update_waveforms(self, ecg_data, ppg_data, rsp_data):
        """Pushes cleaned samples to the sweep plotters."""
        if len(ecg_data) >= 10:
            self.ecg_plotter.add_samples(ecg_data)
        if len(ppg_data) >= 10:
            self.ppg_plotter.add_samples(ppg_data)
        if len(rsp_data) >= 10:
            self.rsp_plotter.add_samples(rsp_data)

    def update_vitals(self, vitals: dict):
        """Updates vitals cards, feature table, and alarm state."""
        # 1. Vital cards
        hr = vitals.get("heart_rate", 0)
        sbp = vitals.get("systolic_bp", 0)
        dbp = vitals.get("diastolic_bp", 0)
        spo2 = vitals.get("spo2", 0)
        rr = vitals.get("respiratory_rate", 0)

        self.hr_card.setValue(f"{hr:.0f}")
        self.bp_card.setValue(f"{sbp:.0f}/{dbp:.0f}")
        if self.bp_card.mean_label:
            map_val = dbp + (sbp - dbp) / 3.0
            self.bp_card.mean_label.setText(f"MAP: {map_val:.0f}")
        self.spo2_card.setValue(f"{spo2:.0f}")
        self.rr_card.setValue(f"{rr:.0f}")

        # Highlight vitals that cross clinical limits
        self.hr_card.checkValue(hr, self.thresh_hr_min, self.thresh_hr_max)
        self.bp_card.checkValue(sbp, self.thresh_sbp_min, self.thresh_sbp_max)
        self.spo2_card.checkValue(spo2, self.thresh_spo2_min, 100.0)
        self.rr_card.checkValue(rr, self.thresh_rr_min, self.thresh_rr_max)

        # Update temperatures
        temp_core = vitals.get("core_temperature", 37.0)
        temp_skin = vitals.get("skin_temperature", 35.5)
        self.lbl_temp_core.setText(f"Core: {temp_core:.2f} °C")
        self.lbl_temp_skin.setText(f"Skin: {temp_skin:.2f} °C")

        # 2. Feature table update
        features = vitals.get("extracted_features", {})
        for row, (key, _, fmt_str, transform) in enumerate(self.FEATURE_DEFS):
            val = features.get(key)
            item = self.feature_table.item(row, 1)
            if item is None:
                continue
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                if transform:
                    val = transform(val)
                try:
                    item.setText(fmt_str.format(val))
                    item.setForeground(QColor("#38bdf8"))
                except Exception:
                    item.setText("ERR")
            else:
                item.setText("—")
                item.setForeground(QColor("#475569"))

        # 3. Alarm banner
        anomaly_score = vitals.get("anomaly_score", 5.0)
        severity = vitals.get("anomaly_severity", "Normal")
        state_name = vitals.get("patient_state", "stable")

        # Check if any vital card is out of range
        is_vital_alert = (
            (hr < self.thresh_hr_min or hr > self.thresh_hr_max) or
            (sbp < self.thresh_sbp_min or sbp > self.thresh_sbp_max) or
            (spo2 < self.thresh_spo2_min) or
            (rr < self.thresh_rr_min or rr > self.thresh_rr_max)
        )

        is_ml_anomaly = (anomaly_score >= self.thresh_ml_threshold) and state_name != "stable"
        active_alarm = is_ml_anomaly or is_vital_alert or self.emergency_active_banner

        if active_alarm:
            if not self.anomaly_active:
                self.anomaly_active = True
                self.flash_timer.start()

            if self.emergency_active_banner:
                self.alarm_banner.setText("⚠  EMERGENCY NURSE CALL ACTIVE  —  CENTRAL STATION NOTIFIED")
            elif is_vital_alert and not is_ml_anomaly:
                self.alarm_banner.setText(
                    f"⚠  VITAL LIMIT EXCEEDED  —  CHECK PATIENT (HR: {hr:.0f}, BP: {sbp:.0f}/{dbp:.0f}, SpO2: {spo2:.0f}%, RR: {rr:.0f})"
                )
            else:
                self.alarm_banner.setText(
                    f"⚠  {severity.upper()} ANOMALY DETECTED ({anomaly_score:.1f})  —  {state_name.upper().replace('_', ' ')}"
                )
        else:
            if self.anomaly_active:
                self.anomaly_active = False
                self.flash_timer.stop()
            self.alarm_banner.setText(
                f"✓  SYSTEM OPERATIONAL (Score: {anomaly_score:.1f}) — PATIENT TELEMETRY ACTIVE"
            )
            self.alarm_banner.setStyleSheet(
                "background-color: rgba(5, 46, 22, 0.8); color: #4ade80;"
                "border: 1px solid #166534; border-radius: 5px;"
            )

    def add_recent_anomaly(self, log_id, state, start_time):
        """Adds a row to the live anomaly feed."""
        t_str = time.strftime("%H:%M:%S", time.localtime(start_time))
        is_stable = state in ("stable", "normal")
        level_str = "INFO" if is_stable else "ALERT"
        color_hex = "#06b6d4" if is_stable else "#ef4444"
        msg = f"ML Detected: {state.upper().replace('_', ' ')}  (Log #{log_id})"

        row_widget = AnomalyFeedRowWidget(t_str, level_str, msg, color_hex)
        item = QListWidgetItem(self.feed_list)
        item.setSizeHint(row_widget.sizeHint())
        self.feed_list.insertItem(0, item)
        self.feed_list.setItemWidget(item, row_widget)

    def update_demo_countdown(self, seconds_left):
        """Shows remaining demo time in the header status badge."""
        mins, secs = divmod(seconds_left, 60)
        self.demo_timer_label.setText(f"⬤  Demo: {mins:02d}:{secs:02d}")
        self.demo_timer_label.setStyleSheet("""
            color: #22d3ee;
            background: rgba(6, 182, 212, 0.1);
            border: 1px solid #0891b2;
            border-radius: 4px;
            padding: 4px 8px;
        """)
        self.inject_btn.setEnabled(False)
        self.anomaly_combo.setEnabled(False)

    def set_active_monitoring_label(self):
        """Resets the header badge to active monitoring state."""
        self.demo_timer_label.setText("⬤  Active Monitoring")
        self.demo_timer_label.setStyleSheet("""
            color: #10b981;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid #10b981;
            border-radius: 4px;
            padding: 4px 8px;
        """)
        self.inject_btn.setEnabled(True)
        self.anomaly_combo.setEnabled(True)

    def _toggle_alarm_flash(self):
        """Alternates alarm banner colors during active anomaly."""
        if not self.anomaly_active:
            return
        self.flash_state = not self.flash_state
        if self.flash_state:
            self.alarm_banner.setStyleSheet(
                "background-color: rgba(127, 29, 29, 0.9); color: #fca5a5;"
                "border: 1px solid #ef4444; border-radius: 5px;"
            )
        else:
            self.alarm_banner.setStyleSheet(
                "background-color: rgba(239, 68, 68, 0.1); color: #ef4444;"
                "border: 1px solid #7f1d1d; border-radius: 5px;"
            )

    def cleanup(self):
        """Stop all running timers — must be called before the window is destroyed."""
        self.flash_timer.stop()
        for plotter in (self.ecg_plotter, self.ppg_plotter, self.rsp_plotter):
            plotter.cleanup()

    def _on_setup_clicked(self):
        """Opens the Alarm Thresholds Configuration Dialog."""
        current_settings = {
            "hr_min": self.thresh_hr_min,
            "hr_max": self.thresh_hr_max,
            "sbp_min": self.thresh_sbp_min,
            "sbp_max": self.thresh_sbp_max,
            "spo2_min": self.thresh_spo2_min,
            "rr_min": self.thresh_rr_min,
            "rr_max": self.thresh_rr_max,
            "ml_threshold": self.thresh_ml_threshold,
            "patient_name": self.patient_name,
            "patient_id": self.patient_id,
            "patient_ward": self.patient_ward,
            "patient_unit": self.patient_unit,
            "patient_bed": self.patient_bed,
            "patient_age": self.patient_age
        }
        
        dialog = SetupDialog(current_settings, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            if dialog.stop_requested:
                self.stop_requested.emit()
            else:
                settings = dialog.get_settings()
                self.thresh_hr_min = settings["hr_min"]
                self.thresh_hr_max = settings["hr_max"]
                self.thresh_sbp_min = settings["sbp_min"]
                self.thresh_sbp_max = settings["sbp_max"]
                self.thresh_spo2_min = settings["spo2_min"]
                self.thresh_rr_min = settings["rr_min"]
                self.thresh_rr_max = settings["rr_max"]
                self.thresh_ml_threshold = settings["ml_threshold"]
                
                # Save patient info
                self.patient_name = settings["patient_name"]
                self.patient_id = settings["patient_id"]
                self.patient_ward = settings["patient_ward"]
                self.patient_unit = settings["patient_unit"]
                self.patient_bed = settings["patient_bed"]
                self.patient_age = settings["patient_age"]
                
                # Update local UI
                self.update_patient_ui()
                
                # Emit signal to notify other components (FeaturesPage)
                self.patient_info_changed.emit({
                    "name": self.patient_name,
                    "id": self.patient_id,
                    "ward": self.patient_ward,
                    "unit": self.patient_unit,
                    "bed": self.patient_bed,
                    "age": self.patient_age
                })
                
                QMessageBox.information(
                    self,
                    "SETTINGS UPDATED",
                    "ICU Patient Information & Alarm Thresholds updated successfully!\n"
                    "Bedside telemetry displays will now update dynamically.",
                    QMessageBox.StandardButton.Ok
                )

    def _on_emergency_clicked(self):
        """Simulates triggering a high-priority nurse call / emergency alert."""
        reply = QMessageBox.question(
            self,
            "TRIGGER EMERGENCY CALL",
            "Are you sure you want to broadcast an emergency distress signal for ICU Bed 12?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            # Emit signal to log to the encrypted database
            self.emergency_triggered.emit()
            
            # Activate visual flashing emergency alarm banner for 15 seconds
            self.emergency_active_banner = True
            if not self.anomaly_active:
                self.anomaly_active = True
                self.flash_timer.start()
            
            # Turn off the emergency banner after 15 seconds
            QTimer.singleShot(15000, self._reset_emergency_banner)
            
            QMessageBox.information(
                self,
                "EMERGENCY CALL ACTIVATED",
                "⚠️ ICU Bed 12 Emergency Call Activated!\n\n"
                "An emergency distress signal has been broadcasted to the Central Nursing Station.\n"
                "ICU Ward A response team has been dispatched immediately.",
                QMessageBox.StandardButton.Ok
            )

    def _reset_emergency_banner(self):
        """Deactivates the emergency banner state."""
        self.emergency_active_banner = False

    def update_patient_ui(self):
        self.lbl_patient.setText(f"<span style='color:#475569;font-size:9px;'>PATIENT</span><br/><b style='color:#e2e8f0;font-size:11px;'>{self.patient_name}</b>")
        self.lbl_location.setText(f"<span style='color:#475569;font-size:9px;'>LOCATION</span><br/><b style='color:#e2e8f0;font-size:11px;'>{self.patient_unit} / {self.patient_bed}</b>")
        self.lbl_ward.setText(f"🏥  {self.patient_ward}")
        self.lbl_unit.setText(f"{self.patient_unit} — {self.patient_bed}")


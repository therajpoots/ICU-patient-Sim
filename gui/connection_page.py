"""
Connection Setup Page for the Bedside Patient Monitor.
Allows configuring remote REST API connections, local simulation, demo modes,
and enabling the local MCP server, styled to match the premium dark-neon HTML design system.
"""

from PyQt6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit, 
    QComboBox, QCheckBox, QPushButton, QFrame, QMessageBox, QApplication
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QFont

class ConnectionPage(QDialog):
    # Signals (kept for backward compatibility, though we use modal results now)
    connect_requested = pyqtSignal(str, str, int, str, str, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ConnectionPage")
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        
        # Initialize public variables to hold clinician choices
        self.selected_mode = "local"
        self.selected_host = ""
        self.selected_port = 8080
        self.selected_api_key = ""
        self.selected_password = ""
        self.selected_mcp_enabled = True
        
        self.init_ui()

    def init_ui(self):
        # Set page-specific dark-neon Tailwind design system stylesheet
        self.setStyleSheet("""
            QDialog#ConnectionPage, QWidget#ConnectionPage {
                background-color: transparent;
            }
            QFrame#FormCard {
                background-color: #0c1524; /* bg-surface-container */
                border: 1px solid rgba(34, 211, 238, 0.25); /* cyan glow border */
                border-radius: 16px;
            }
            QLabel#MainTitle {
                color: #ffffff; /* on-surface */
                font-family: 'Inter', sans-serif;
                font-size: 30px;
                font-weight: 800;
                letter-spacing: -0.5px;
                qproperty-alignment: 'AlignCenter';
            }
            QLabel#SubTitle {
                color: #64748b; /* on-surface-variant */
                font-family: 'JetBrains Mono', sans-serif;
                font-size: 9px;
                font-weight: 800;
                letter-spacing: 2px;
                qproperty-alignment: 'AlignCenter';
            }
            QLabel#FormLabel {
                color: #94a3b8; /* label-caps style */
                font-family: 'JetBrains Mono', sans-serif;
                font-size: 10px;
                font-weight: bold;
                text-transform: uppercase;
                margin-bottom: 2px;
            }
            QLabel#NetworkHeader {
                color: #22d3ee; /* tertiary */
                font-family: 'JetBrains Mono', sans-serif;
                font-size: 10px;
                font-weight: bold;
                text-transform: uppercase;
            }
            QLineEdit, QComboBox {
                background-color: #05080f; /* bg-surface-container-high */
                border: 1px solid #1e2d45;
                border-radius: 6px;
                padding: 9px 12px;
                color: #f8fafc;
                font-family: 'JetBrains Mono';
                font-size: 12px;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 1px solid #22d3ee; /* focus:border-tertiary */
            }
            QCheckBox {
                color: #cbd5e1;
                font-family: 'JetBrains Mono';
                font-size: 11px;
            }
            QFrame#NetworkFrame {
                background-color: #04070d; /* bg-surface-container-lowest */
                border: 1px solid #1e2d45;
                border-radius: 8px;
            }
            QPushButton#VisibilityBtn {
                background-color: #05080f;
                border: 1px solid #1e2d45;
                border-radius: 6px;
                padding: 8px;
                color: #94a3b8;
                min-width: 32px;
            }
            QPushButton#VisibilityBtn:hover {
                background-color: #0c1524;
            }
            QFrame#ActionsFrame {
                background-color: #080f1d; /* bg-surface-container-highest */
                border-top: 1px solid #1e2d45;
                border-bottom-left-radius: 16px;
                border-bottom-right-radius: 16px;
            }
            QPushButton#TestBtn {
                background-color: transparent;
                color: #38bdf8;
                border: 1px solid rgba(56, 189, 248, 0.3);
                border-radius: 6px;
                padding: 11px 20px;
                font-family: 'JetBrains Mono';
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton#TestBtn:hover {
                background-color: rgba(56, 189, 248, 0.1);
            }
            QPushButton#StartBtn {
                background-color: #06b6d4; /* bg-tertiary */
                color: #ffffff; /* on-tertiary */
                border: none;
                border-radius: 6px;
                padding: 11px 24px;
                font-family: 'JetBrains Mono';
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton#StartBtn:hover {
                background-color: #22d3ee; /* hover:bg-tertiary-fixed */
            }
            QPushButton#CloseWindowBtn {
                background-color: transparent;
                color: #c6c6cd;
                border: none;
                font-family: 'Inter', sans-serif;
                font-size: 22px;
                font-weight: bold;
            }
            QPushButton#CloseWindowBtn:hover {
                color: #ffb4ab;
            }
        """)

        # Main Layout - stretch FormCard to occupy full height
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Center container for the form (Tailwind Mockup style)
        form_card = QFrame()
        form_card.setObjectName("FormCard")
        form_card.setFixedWidth(450)
        
        card_layout = QVBoxLayout(form_card)
        card_layout.setContentsMargins(0, 0, 0, 0)
        card_layout.setSpacing(0)
        
        # Fields Container (tighter vertical padding and spacing for 600px fit)
        fields_container = QWidget()
        fields_layout = QVBoxLayout(fields_container)
        fields_layout.setContentsMargins(30, 15, 30, 10)
        fields_layout.setSpacing(12)

        # Top close button layout
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 0, 0)
        close_btn = QPushButton("×")
        close_btn.setObjectName("CloseWindowBtn")
        close_btn.setFixedSize(30, 30)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(self.close_app)
        top_bar.addStretch()
        top_bar.addWidget(close_btn)
        fields_layout.addLayout(top_bar)
        
        # 1. Header Section
        header_layout = QVBoxLayout()
        header_layout.setSpacing(10)
        header_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Ambient Badge
        badge = QLabel("💙")
        badge.setFont(QFont("Inter", 24))
        badge.setFixedSize(64, 64)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setStyleSheet("""
            QLabel {
                background-color: rgba(34, 211, 238, 0.1);
                border: 1px solid rgba(34, 211, 238, 0.35);
                border-radius: 32px;
            }
        """)
        header_layout.addWidget(badge, alignment=Qt.AlignmentFlag.AlignCenter)
        
        title_label = QLabel("HealthFi")
        title_label.setObjectName("MainTitle")
        
        subtitle_label = QLabel("CLINICAL BEDSIDE TELEMETRY MONITOR")
        subtitle_label.setObjectName("SubTitle")
        
        header_layout.addWidget(title_label)
        header_layout.addWidget(subtitle_label)
        fields_layout.addLayout(header_layout)
        
        # 2. Telemetry Source Mode Selection
        mode_box = QVBoxLayout()
        mode_box.setSpacing(4)
        
        mode_label = QLabel("Telemetry Source Mode")
        mode_label.setObjectName("FormLabel")
        
        self.mode_combo = QComboBox()
        self.mode_combo.setObjectName("FormInputCombo")
        self.mode_combo.addItems(["Simulate Patient (Local)", "Demo Mode (10-Min / 6 Anomalies)", "Remote Patient Monitor (REST SSE)"])
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        
        mode_helper = QLabel("Select the data ingestion pipeline for real-time waveform rendering.")
        mode_helper.setStyleSheet("color: #a1a1aa; font-family: 'JetBrains Mono'; font-size: 10px;")
        
        mode_box.addWidget(mode_label)
        mode_box.addWidget(self.mode_combo)
        mode_box.addWidget(mode_helper)
        fields_layout.addLayout(mode_box)
        
        # 3. Network Configuration Group (Container Frame)
        self.network_widget = QFrame()
        self.network_widget.setObjectName("NetworkFrame")
        
        net_layout = QVBoxLayout(self.network_widget)
        net_layout.setContentsMargins(20, 20, 20, 20)
        net_layout.setSpacing(15)
        
        # Net Configuration Header
        net_hdr_layout = QHBoxLayout()
        net_hdr_layout.setSpacing(8)
        net_icon = QLabel("🌐")
        net_icon.setFont(QFont("Inter", 11))
        net_title = QLabel("Network Configuration")
        net_title.setObjectName("NetworkHeader")
        net_hdr_layout.addWidget(net_icon)
        net_hdr_layout.addWidget(net_title)
        net_hdr_layout.addStretch()
        net_layout.addLayout(net_hdr_layout)
        
        # Grid layout for IP and Port
        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setSpacing(15)
        
        host_lbl = QLabel("Remote Host IP Address")
        host_lbl.setObjectName("FormLabel")
        self.host_input = QLineEdit("127.0.0.1")
        self.host_input.setPlaceholderText("e.g. 192.168.1.100")
        
        port_lbl = QLabel("Remote API Port")
        port_lbl.setObjectName("FormLabel")
        self.port_input = QLineEdit("8080")
        self.port_input.setPlaceholderText("e.g. 8080")
        
        grid_layout.addWidget(host_lbl, 0, 0)
        grid_layout.addWidget(self.host_input, 1, 0)
        grid_layout.addWidget(port_lbl, 0, 1)
        grid_layout.addWidget(self.port_input, 1, 1)
        net_layout.addWidget(grid_widget)
        
        # API Key Row
        api_box = QVBoxLayout()
        api_box.setSpacing(4)
        api_lbl = QLabel("DeepSeek API Key / Authentication Key")
        api_lbl.setObjectName("FormLabel")
        
        api_input_layout = QHBoxLayout()
        api_input_layout.setSpacing(8)
        
        self.api_input = QLineEdit("sk-3ae47177f18e4ecf808440d6168c0d6f")
        self.api_input.setPlaceholderText("Enter API Key")
        self.api_input.setEchoMode(QLineEdit.EchoMode.Password)
        
        self.api_visibility_btn = QPushButton("👁")
        self.api_visibility_btn.setObjectName("VisibilityBtn")
        self.api_visibility_btn.setCheckable(True)
        self.api_visibility_btn.clicked.connect(self.toggle_api_visibility)
        
        api_input_layout.addWidget(self.api_input, stretch=1)
        api_input_layout.addWidget(self.api_visibility_btn)
        
        api_box.addWidget(api_lbl)
        api_box.addLayout(api_input_layout)
        net_layout.addLayout(api_box)
        
        fields_layout.addWidget(self.network_widget)
        self.network_widget.setVisible(False) # Local simulation by default
        
        # 4. Security Password Row
        sec_box = QVBoxLayout()
        sec_box.setSpacing(4)
        
        sec_lbl = QLabel("Clinician Security Access Key <font color='#06b6d4'>🔒</font>")
        sec_lbl.setObjectName("FormLabel")
        
        self.pwd_input = QLineEdit()
        self.pwd_input.setPlaceholderText("Enter secure DB decryption key")
        self.pwd_input.setEchoMode(QLineEdit.EchoMode.PasswordEchoOnEdit)
        
        sec_helper = QLabel("Secures local patient telemetry logs. Uses PBKDF2-HMAC-SHA256 key derivation with a high iteration count.")
        sec_helper.setWordWrap(True)
        sec_helper.setStyleSheet("color: #a1a1aa; font-family: 'JetBrains Mono'; font-size: 10px; line-height: 1.2;")
        
        sec_box.addWidget(sec_lbl)
        sec_box.addWidget(self.pwd_input)
        sec_box.addWidget(sec_helper)
        fields_layout.addLayout(sec_box)
        
        # 5. MCP Checkbox option box
        chk_container = QFrame()
        chk_container.setStyleSheet("QFrame { background-color: #05080f; border: 1px solid #1e2d45; border-radius: 6px; }")
        chk_layout = QHBoxLayout(chk_container)
        chk_layout.setContentsMargins(15, 12, 15, 12)
        chk_layout.setSpacing(12)
        
        self.mcp_checkbox = QCheckBox("Enable Local MCP Server (Port 8000)")
        self.mcp_checkbox.setChecked(True)
        
        chk_desc_layout = QVBoxLayout()
        chk_desc_layout.setSpacing(2)
        chk_desc_layout.addWidget(self.mcp_checkbox)
        
        chk_desc = QLabel("Allows secure Model Context Protocol interactions for advanced AI diagnostics.")
        chk_desc.setStyleSheet("color: #a1a1aa; font-family: 'JetBrains Mono'; font-size: 10px; border: none; background: transparent;")
        chk_desc_layout.addWidget(chk_desc)
        
        chk_layout.addLayout(chk_desc_layout)
        fields_layout.addWidget(chk_container)
        
        card_layout.addWidget(fields_container)
        
        # 6. Action Footer (px-8 py-6 bg-surface-container-highest)
        actions_frame = QFrame()
        actions_frame.setObjectName("ActionsFrame")
        actions_layout = QHBoxLayout(actions_frame)
        actions_layout.setContentsMargins(30, 15, 30, 15)
        actions_layout.setSpacing(15)
        
        self.test_btn = QPushButton("Test Telemetry Stream")
        self.test_btn.setObjectName("TestBtn")
        self.test_btn.clicked.connect(self.on_test_connection)
        
        self.start_btn = QPushButton("Start Clinical Monitoring")
        self.start_btn.setObjectName("StartBtn")
        self.start_btn.clicked.connect(self.on_start_monitoring)
        
        actions_layout.addWidget(self.test_btn)
        actions_layout.addWidget(self.start_btn)
        self.test_btn.setVisible(False) # Hidden by default since Local Simulation is selected initially
        
        card_layout.addWidget(actions_frame)
        
        layout.addWidget(form_card)
        self.setLayout(layout)
        
        # Initialize sizing and modes
        self.on_mode_changed(0)

    def toggle_api_visibility(self, checked):
        """Toggles the password/plaintext visibility on the API key field."""
        if checked:
            self.api_input.setEchoMode(QLineEdit.EchoMode.Normal)
            self.api_visibility_btn.setText("🙈")
        else:
            self.api_input.setEchoMode(QLineEdit.EchoMode.Password)
            self.api_visibility_btn.setText("👁")

    def on_mode_changed(self, index):
        """Shows or hides network fields based on selected mode."""
        is_remote = (index == 2)
        self.network_widget.setVisible(is_remote)
        self.test_btn.setVisible(is_remote)
        
        if is_remote:
            self.port_input.setText("8080")
            self.test_btn.setEnabled(True)
            self.setFixedSize(450, 710)
        else:
            self.test_btn.setEnabled(False)
            self.setFixedSize(450, 530)
            
        self.center_on_screen()

    def on_test_connection(self):
        """Test API connection to remote patient monitor."""
        host = self.host_input.text().strip()
        port_str = self.port_input.text().strip()
        api_key = self.api_input.text().strip()
        
        if not host or not port_str:
            QMessageBox.critical(self, "Validation Error", "Please input host and port.")
            return
            
        try:
            port = int(port_str)
        except ValueError:
            QMessageBox.critical(self, "Validation Error", "Port must be an integer.")
            return

        import httpx
        self.test_btn.setText("Connecting...")
        self.test_btn.setEnabled(False)
        self.repaint()
        
        import logging
        logger = logging.getLogger("ConnectionPage")
        try:
            headers = {"X-API-Key": api_key}
            url = f"http://{host}:{port}/api/v1/vitals/current"
            r = httpx.get(url, headers=headers, timeout=2.0)
            if r.status_code == 200:
                QMessageBox.information(self, "Success", "Connection established! Handshake completed.")
            else:
                QMessageBox.warning(self, "Warning", f"Received status code {r.status_code} from endpoint.")
        except Exception as e:
            logger.exception("Connection test failure:")
            QMessageBox.critical(self, "Connection Failed", f"Could not reach remote patient monitor:\n{e}")
        finally:
            self.test_btn.setText("Test Telemetry Stream")
            self.test_btn.setEnabled(True)

    def on_start_monitoring(self):
        """Validates input parameters and requests connection start."""
        idx = self.mode_combo.currentIndex()
        modes = ["local", "demo", "remote"]
        self.selected_mode = modes[idx]
        
        self.selected_host = self.host_input.text().strip()
        port_str = self.port_input.text().strip()
        self.selected_api_key = self.api_input.text().strip()
        self.selected_password = self.pwd_input.text()
        self.selected_mcp_enabled = self.mcp_checkbox.isChecked()
        
        if not self.selected_password:
            QMessageBox.critical(self, "Validation Error", "An encryption password is required to initialize/decrypt the local database.")
            return
            
        self.selected_port = 8080
        if self.selected_mode == "remote":
            if not self.selected_host or not port_str:
                QMessageBox.critical(self, "Validation Error", "Host and Port are required for remote monitoring.")
                return
            try:
                self.selected_port = int(port_str)
            except ValueError:
                QMessageBox.critical(self, "Validation Error", "Port must be an integer.")
                return
                
        # Also emit the signal in case anything external relies on it
        self.connect_requested.emit(self.selected_mode, self.selected_host, self.selected_port, self.selected_api_key, self.selected_password, self.selected_mcp_enabled)
        self.accept()

    def close_app(self):
        """Closes the dialog with rejection."""
        self.reject()

    def center_on_screen(self):
        """Centers the connection dialog on the screen."""
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            x = (geo.width() - self.width()) // 2
            y = (geo.height() - self.height()) // 2
            y = max(0, y - 25)
            self.move(x, y)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton and hasattr(self, "_drag_position"):
            self.move(event.globalPosition().toPoint() - self._drag_position)
            event.accept()

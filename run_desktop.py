"""
Desktop ICU Bedside Patient Monitor Application Entry Runner.
Instantiates QApplication and displays the PatientMonitorApp main window.
"""

import sys
import os
import faulthandler

# Enable crash logging to file (captures native crash stack traces)
os.makedirs("data", exist_ok=True)
_crash_log = open(os.path.join("data", "crash.log"), "w")
faulthandler.enable(_crash_log)

from PyQt6.QtWidgets import QApplication, QDialog
from PyQt6.QtCore import Qt
from gui.app import PatientMonitorApp
from gui.connection_page import ConnectionPage


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    dialog = ConnectionPage()
    if dialog.exec() == QDialog.DialogCode.Accepted:
        window = PatientMonitorApp()
        window.start_monitoring(
            mode=dialog.selected_mode,
            host=dialog.selected_host,
            port=dialog.selected_port,
            api_key=dialog.selected_api_key,
            password=dialog.selected_password,
            mcp_enabled=dialog.selected_mcp_enabled
        )
        window.show()
        exit_code = app.exec()
    else:
        exit_code = 0

    _crash_log.close()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

import os

from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QDialog

from pin_lock.security_manager import SecurityManager
from pin_lock.pin_dialog import PinLockDialog
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings

import sys
from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Consistent styling
    security_manager = SecurityManager()
    
    if not security_manager.is_pin_set():
        # First time run: force user to set a PIN
        dialog = PinLockDialog(security_manager, mode='set')
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return -1 # User cancelled setting a PIN, exit app
    
    # Subsequent runs: verify the PIN
    login_dialog = PinLockDialog(security_manager, mode='verify')
    if login_dialog.exec() == QDialog.DialogCode.Accepted:
        # If the PIN was correct, show the main window
        main_window = MainWindow()
        main_window.show()
        sys.exit(app.exec())
    else:
        # User cancelled login or failed too many times
        sys.exit(0)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
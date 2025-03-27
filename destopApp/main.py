import sys
from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow

def main():
    """Entry point for the OMR Scanner application"""
    # Create the Qt application
    app = QApplication(sys.argv)
    
    # Apply basic styling (optional)
    app.setStyle("Fusion")
    
    # Create and show the main window
    window = MainWindow()
    window.show()
    
    # Start the Qt event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
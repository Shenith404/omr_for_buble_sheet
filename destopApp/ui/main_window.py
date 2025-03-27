from PySide6.QtWidgets import (
    QMainWindow, 
    QWidget, 
    QHBoxLayout, 
    QVBoxLayout,
    QTabWidget, 
    QListWidget, 
    QStackedWidget,
    QGroupBox  # Ensure all needed widgets are imported
)
from PySide6.QtCore import Qt
# To:
from ui.processing_tab import ProcessingTab
from ui.results_tab import ResultsTab
from ui.settings_tab import SettingsTab
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OMR Scanner Pro")
        self.resize(1200, 800)
        
        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main Layout (Sidebar + Content)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # --- Sidebar ---
        self.sidebar = QListWidget()
        self.sidebar.setFixedWidth(200)
        self.sidebar.addItems(["Processing", "Results", "Settings"])
        self.sidebar.currentRowChanged.connect(self.switch_tab)
        main_layout.addWidget(self.sidebar)
        
        # --- Content Area ---
        self.content_stack = QStackedWidget()
        
        # Create tabs
        self.processing_tab = ProcessingTab()
        self.results_tab = ResultsTab()
        self.settings_tab = SettingsTab()
        
        self.content_stack.addWidget(self.processing_tab)
        self.content_stack.addWidget(self.results_tab)
        self.content_stack.addWidget(self.settings_tab)
        
        main_layout.addWidget(self.content_stack, stretch=1)
        
    def switch_tab(self, index):
        self.content_stack.setCurrentIndex(index)
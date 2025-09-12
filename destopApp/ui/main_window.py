import os
import csv
import cv2
from PySide6.QtWidgets import (
    QMainWindow, QTabWidget, QStatusBar,
    QMessageBox, QVBoxLayout, QWidget,
    QMenuBar, QMenu, QFileDialog, QLabel, QProgressBar
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from ui.project_tab import ProjectTab
from ui.processing_tab import ProcessingTab
from ui.review_tab import ReviewTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OMR Scanner Pro • Professional Answer Sheet Analysis")
        self.resize(1400, 900)

        # App state
        self.current_project = None
        self.current_images = []
        self.linked_images = False

        # Tabs
        self.project_tab = ProjectTab()
        self.processing_tab = ProcessingTab()
        self.review_tab = ReviewTab()

        # Setup UI
        self.setup_ui()
        self.setup_connections()

        # Disable until project is active
        self.tab_widget.setTabEnabled(1, False)
        self.tab_widget.setTabEnabled(2, False)

        self.update_ui_state()

        # Apply dark theme
        self.apply_dark_theme()


    def setup_ui(self):
        # Menu bar
        self.setup_menu()

        # Central layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(8)

        # Tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)

        self.tab_widget.addTab(self.project_tab, "Project")
        self.tab_widget.addTab(self.processing_tab, "Processing")
        self.tab_widget.addTab(self.review_tab, "Review")

        main_layout.addWidget(self.tab_widget)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("Ready to create or open a project")
        self.status_bar.addPermanentWidget(self.status_label, 1)

        self.progress = QProgressBar()
        self.progress.setFixedWidth(200)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress)


    def apply_dark_theme(self):
        """Professional dark theme inspired by VS Code & JetBrains IDEs"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #181818;
                color: #e8e8e8;
            }
            
            QWidget {
                background-color: #181818;
                color: #e8e8e8;
                font-size: 13px;
                font-family: 'Segoe UI', 'SF Pro Display', 'Inter', 'Roboto', sans-serif;
                font-weight: 400;
            }
            
            /* Tab Widget Styling */
            QTabWidget::pane {
                border: 1px solid #3c3c3c;
                background: #1e1e1e;
                border-radius: 8px;
                margin-top: -1px;
            }
            
            QTabBar::tab {
                background: linear-gradient(180deg, #2d2d2d 0%, #262626 100%);
                color: #cccccc;
                border: 1px solid #3c3c3c;
                border-bottom: none;
                padding: 12px 24px;
                margin-right: 1px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                min-width: 120px;
                font-weight: 500;
            }
            
            QTabBar::tab:selected {
                background: linear-gradient(180deg, #1e1e1e 0%, #1a1a1a 100%);
                border-top: 2px solid #007acc;
                color: #ffffff;
                font-weight: 600;
                border-left: 1px solid #007acc;
                border-right: 1px solid #007acc;
            }
            
            QTabBar::tab:hover:!selected {
                background: linear-gradient(180deg, #383838 0%, #323232 100%);
                color: #ffffff;
            }
            
            QTabBar::tab:first {
                margin-left: 0;
            }
            
            /* Status Bar */
            QStatusBar {
                background: linear-gradient(180deg, #1c1c1c 0%, #161616 100%);
                border-top: 1px solid #3c3c3c;
                color: #a0a0a0;
                font-size: 12px;
                font-weight: 400;
                padding: 4px 8px;
            }
            
            QStatusBar QLabel {
                color: #a0a0a0;
                padding: 2px 8px;
            }
            
            /* Menu Bar */
            QMenuBar {
                background: linear-gradient(180deg, #2a2a2a 0%, #242424 100%);
                color: #e8e8e8;
                font-size: 13px;
                font-weight: 500;
                border-bottom: 1px solid #3c3c3c;
                padding: 2px;
            }
            
            QMenuBar::item {
                background: transparent;
                padding: 8px 16px;
                border-radius: 4px;
                margin: 2px;
            }
            
            QMenuBar::item:selected {
                background: linear-gradient(180deg, #0078d4 0%, #005a9e 100%);
                color: #ffffff;
            }
            
            QMenuBar::item:pressed {
                background: #004578;
            }
            
            /* Menu Dropdown */
            QMenu {
                background: #2a2a2a;
                border: 1px solid #404040;
                border-radius: 6px;
                color: #e8e8e8;
                padding: 4px 0;
            }
            
            QMenu::item {
                padding: 8px 16px 8px 24px;
                margin: 1px 4px;
                border-radius: 4px;
                font-weight: 400;
            }
            
            QMenu::item:selected {
                background: linear-gradient(180deg, #0078d4 0%, #005a9e 100%);
                color: #ffffff;
            }
            
            QMenu::separator {
                height: 1px;
                background: #404040;
                margin: 4px 8px;
            }
            
            /* Progress Bar */
            QProgressBar {
                border: 2px solid #404040;
                border-radius: 8px;
                text-align: center;
                font-size: 12px;
                font-weight: 600;
                background-color: #2a2a2a;
                color: #ffffff;
                height: 20px;
                padding: 1px;
            }
            
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0078d4, stop:0.5 #00bcf2, stop:1 #17a2b8);
                border-radius: 6px;
                margin: 1px;
                min-width: 8px;
            }
            
            QProgressBar[value="0"] {
                color: #888888;
            }
            
            /* General Labels */
            QLabel {
                color: #e8e8e8;
                font-weight: 400;
            }
            
            /* Group Boxes */
            QGroupBox {
                background: #1e1e1e;
                border: 1px solid #404040;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 600;
                color: #e8e8e8;
                margin-top: 12px;
                padding-top: 8px;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px 0 8px;
                color: #ffffff;
                background: #1e1e1e;
            }
            
            /* Buttons */
            QPushButton {
                background: linear-gradient(180deg, #0078d4 0%, #005a9e 100%);
                border: 1px solid #004578;
                border-radius: 6px;
                color: #ffffff;
                font-size: 13px;
                font-weight: 500;
                padding: 10px 20px;
                min-width: 100px;
                min-height: 16px;
            }
            
            QPushButton:hover {
                background: linear-gradient(180deg, #1084d8 0%, #0066b2 100%);
                border-color: #0078d4;
            }
            
            QPushButton:pressed {
                background: linear-gradient(180deg, #005a9e 0%, #004578 100%);
                border-color: #003d69;
            }
            
            QPushButton:disabled {
                background: #3a3a3a;
                border-color: #525252;
                color: #7a7a7a;
            }
            
            /* Input Fields */
            QLineEdit, QComboBox {
                background: #2a2a2a;
                border: 1px solid #404040;
                border-radius: 6px;
                color: #e8e8e8;
                font-size: 13px;
                padding: 8px 12px;
                selection-background-color: #0078d4;
            }
            
            QLineEdit:focus, QComboBox:focus {
                border-color: #0078d4;
                background: #323232;
            }
            
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 4px solid #e8e8e8;
                margin-right: 8px;
            }
            
            QComboBox QAbstractItemView {
                background: #2a2a2a;
                border: 1px solid #404040;
                border-radius: 6px;
                color: #e8e8e8;
                selection-background-color: #0078d4;
                outline: none;
            }
            
            /* Scrollbars */
            QScrollBar:vertical {
                background: #2a2a2a;
                width: 12px;
                border-radius: 6px;
                margin: 0;
            }
            
            QScrollBar::handle:vertical {
                background: #525252;
                border-radius: 6px;
                min-height: 30px;
            }
            
            QScrollBar::handle:vertical:hover {
                background: #6a6a6a;
            }
            
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
        """)


    def update_ui_state(self):
        """Update UI elements based on current state"""
        if self.current_project:
            self.tab_widget.setTabEnabled(1, True)
            self.tab_widget.setTabEnabled(2, True)

            msg = f"Project: {os.path.basename(self.current_project)}"
            if self.current_images:
                msg += f" | {len(self.current_images)} {'linked' if self.linked_images else 'loaded'} images"
            self.status_label.setText(msg)
        else:
            self.tab_widget.setTabEnabled(1, False)
            self.tab_widget.setTabEnabled(2, False)
            if self.tab_widget.currentIndex() in [1, 2]:
                self.tab_widget.setCurrentIndex(0)
            self.status_label.setText("Ready to create or open a project")



    def setup_menu(self):
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        new_project = file_menu.addAction("New Project")
        new_project.triggered.connect(lambda: self.tab_widget.setCurrentIndex(0))
        
        open_project = file_menu.addAction("Open Project")
        open_project.triggered.connect(self.open_project)
        
        file_menu.addSeparator()
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)


    def setup_connections(self):
        # Project tab signals
        self.project_tab.project_created.connect(self.handle_project_created)
        self.project_tab.project_opened.connect(self.handle_project_opened)
        self.project_tab.images_added.connect(self.handle_images_added)
        
        # Processing tab signals
        self.processing_tab.processing_complete.connect(self.handle_processing_complete)
        self.processing_tab.processing_cancelled.connect(self.handle_processing_cancelled)
        self.processing_tab.processing_started.connect(self.handle_processing_started)
        self.processing_tab.processing_finished.connect(self.handle_processing_finished)
        
        # Tab change event
        self.tab_widget.currentChanged.connect(self.handle_tab_changed)


    def handle_tab_changed(self, index):
        """Force user back to Project tab if no project exists"""
        if not self.current_project and index in [1, 2]:
            QMessageBox.warning(self, "No Project", "Please create or open a project first.")
            self.tab_widget.setCurrentIndex(0)


    def open_project(self):
        """Open an existing project with validation"""
        project_path = QFileDialog.getExistingDirectory(self, "Open Project")
        if project_path:
            if not os.path.exists(os.path.join(project_path, "image_references.txt")) and \
               not os.path.exists(os.path.join(project_path, "original_images")):
                QMessageBox.warning(self, "Invalid Project", "Selected folder is not a valid OMR project")
                return
            
            self.project_tab.load_project(project_path)


    def handle_project_created(self, project_path):
        """Handle new project creation"""
        self.current_project = project_path
        self.current_images = []
        self.linked_images = False
        self.processing_tab.load_project(project_path)
        self.review_tab.load_project(project_path)
        self.update_ui_state()
        
        QMessageBox.information(
            self, 
            "Project Created", 
            f"Project created successfully at:\n{project_path}"
        )


    def handle_project_opened(self, project_path):
        """Handle opening an existing project"""
        self.current_project = project_path
        self.current_images = []
        self.processing_tab.load_project(project_path)
        self.review_tab.load_project(project_path)
        self.update_ui_state()
        self.status_bar.showMessage(f"Opened project: {os.path.basename(project_path)}")


    def handle_images_added(self, image_paths):
        """Handle new images added to project"""
        self.current_images = image_paths
        self.linked_images = all(not path.startswith(self.current_project) for path in image_paths)
        
        # Debug print
        print(f"MainWindow received {len(image_paths)} images:")
        for path in image_paths:
            print(f" - {path} (exists: {os.path.exists(path)})")
        
        self.processing_tab.load_project(self.current_project)
        self.processing_tab.set_image_paths(image_paths.copy())
        self.review_tab.load_images(image_paths)
        
        self.update_ui_state()


    def handle_processing_started(self):
        self.tab_widget.setTabEnabled(0, False)
        self.status_bar.showMessage("Processing started...")


    def handle_processing_finished(self):
        self.tab_widget.setTabEnabled(0, True)
        self.status_bar.showMessage("Processing finished")


    def handle_processing_complete(self, all_answers, processed_images, debug_imgs):
        """Handle completed batch processing"""
        if self.current_project:
            results_dir = os.path.join(self.current_project, "results")
            os.makedirs(results_dir, exist_ok=True)
            
            # Save CSV
            csv_path = os.path.join(results_dir, "answers.csv")
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Image"] + [f"Q{i+1}" for i in range(len(all_answers[0]))])
                for i, answers in enumerate(all_answers):
                    img_name = os.path.basename(self.current_images[i])
                    writer.writerow([img_name] + answers)
            
            # Save processed images
            for i, img in enumerate(processed_images):
                img_name = os.path.basename(self.current_images[i])
                cv2.imwrite(os.path.join(results_dir, f"processed_{img_name}"), img)
            
            self.status_bar.showMessage(f"Processing complete! Results saved to {results_dir}")
            QMessageBox.information(
                self,
                "Processing Complete",
                f"Processed {len(all_answers)} images\n"
                f"Results saved to:\n{results_dir}"
            )


    def handle_processing_cancelled(self):
        self.status_bar.showMessage("Processing cancelled by user")


    def closeEvent(self, event):
        """Handle window close event with cleanup"""
        if hasattr(self.processing_tab, 'processing') and self.processing_tab.processing:
            reply = QMessageBox.question(
                self,
                "Processing Active",
                "A processing operation is still running. Are you sure you want to quit?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                event.ignore()
                return
        
        if hasattr(self.project_tab, 'webcam_active') and self.project_tab.webcam_active:
            self.project_tab.stop_webcam()
        
        if hasattr(self.processing_tab, 'cap') and self.processing_tab.cap:
            self.processing_tab.cap.release()
        
        event.accept()

import os
import csv
import cv2
from PySide6.QtWidgets import (
    QMainWindow, QTabWidget, QStatusBar, 
    QMessageBox, QVBoxLayout, QWidget, 
    QMenuBar, QMenu
)
from PySide6.QtCore import Qt, Signal
from ui.project_tab import ProjectTab
from ui.processing_tab import ProcessingTab

class MainWindow(QMainWindow):
    def __init__(self):
            super().__init__()
            self.setWindowTitle("OMR Scanner Pro")
            self.resize(1200, 800)
            
            # Initialize application state
            self.current_project = None
            self.current_images = []
            self.linked_images = False
            
            # Initialize tabs
            self.project_tab = ProjectTab()
            self.processing_tab = ProcessingTab()
            
            # Setup UI
            self.setup_ui()
            self.setup_connections()
            
            # Ensure all tabs are enabled
            self.tab_widget.setTabEnabled(0, True)
            self.tab_widget.setTabEnabled(1, True)


    def setup_ui(self):
        # Menu bar
        self.setup_menu()
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)
        
        # Add tabs
        self.tab_widget.addTab(self.project_tab, "📁 Project")
        self.tab_widget.addTab(self.processing_tab, "🔍 Processing")
        
        # Remove the tab restriction
        # self.tab_widget.setTabEnabled(1, False)  # Comment out or remove this line
        
        main_layout.addWidget(self.tab_widget)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready to create or open a project")

    def update_ui_state(self):
        """Update UI elements based on current state"""
        # This method can be simplified or removed if not needed
        if self.current_project:
            msg = f"Project: {os.path.basename(self.current_project)}"
            if self.current_images:
                msg += f" | {len(self.current_images)} {'linked' if self.linked_images else 'loaded'} images"
            self.status_bar.showMessage(msg)
        else:
            self.status_bar.showMessage("Ready to create or open a project")

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
        
        # Tab change events
        self.tab_widget.currentChanged.connect(self.handle_tab_changed)

    def handle_tab_changed(self, index):
        """Strictly enforce project requirements before allowing processing tab access"""
        # if index == 1:  # Processing tab selected
        #     if not self.current_project:
        #         QMessageBox.warning(
        #             self,
        #             "No Project",
        #             "Please create or open a project first",
        #             QMessageBox.Ok
        #         )
        #         self.tab_widget.setCurrentIndex(0)
        #         return
            
        #     if not self.current_images:
        #         QMessageBox.warning(
        #             self,
        #             "No Images",
        #             "Please add images to the project before processing",
        #             QMessageBox.Ok
        #         )
        #         self.tab_widget.setCurrentIndex(0)
        #         return
        pass

    def open_project(self):
        """Open an existing project with validation"""
        project_path = QFileDialog.getExistingDirectory(self, "Open Project")
        if project_path:
            # Verify it's a valid project directory
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
        self.update_ui_state()
        self.status_bar.showMessage(f"Opened project: {os.path.basename(project_path)}")

   # In MainWindow's handle_images_added method:
    def handle_images_added(self, image_paths):
        """Handle new images added to project"""
        self.current_images = image_paths
        self.linked_images = all(not path.startswith(self.current_project) for path in image_paths)
        
        # Debug print to verify images
        print(f"MainWindow received {len(image_paths)} images:")
        for path in image_paths:
            print(f" - {path} (exists: {os.path.exists(path)})")
        
        # Properly update processing tab
        self.processing_tab.load_project(self.current_project)
        self.processing_tab.set_image_paths(image_paths.copy())  # Use copy to avoid reference issues
        
        self.update_ui_state()
        self.tab_widget.setTabEnabled(1, True)  # Ensure processing tab is enabled

        
    def handle_processing_started(self):
        """Disable UI elements during processing"""
        self.tab_widget.setTabEnabled(0, False)  # Disable project tab
        self.status_bar.showMessage("Processing started...")

    def handle_processing_finished(self):
        """Re-enable UI elements after processing"""
        self.tab_widget.setTabEnabled(0, True)  # Enable project tab
        self.status_bar.showMessage("Processing finished")

    def handle_processing_complete(self, all_answers, processed_images, debug_imgs):
        """Handle completed batch processing"""
        if self.current_project:
            results_dir = os.path.join(self.current_project, "results")
            os.makedirs(results_dir, exist_ok=True)
            
            # Save all answers to CSV
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

    def update_ui_state(self):
        """Update UI elements based on current state"""
        # Always enable both tabs
        self.tab_widget.setTabEnabled(0, True)  # Project tab
        self.tab_widget.setTabEnabled(1, True)  # Processing tab
        
        # Update status bar message (optional)
        if self.current_project:
            msg = f"Project: {os.path.basename(self.current_project)}"
            if self.current_images:
                msg += f" | {len(self.current_images)} {'linked' if self.linked_images else 'loaded'} images"
            self.status_bar.showMessage(msg)
        else:
            self.status_bar.showMessage("Ready to create or open a project")

    def closeEvent(self, event):
        """Handle window close event with proper cleanup"""
        # Check for active processing
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
        
        # Clean up webcam resources if active
        if hasattr(self.project_tab, 'webcam_active') and self.project_tab.webcam_active:
            self.project_tab.stop_webcam()
        
        # Clean up processing tab resources
        if hasattr(self.processing_tab, 'cap') and self.processing_tab.cap:
            self.processing_tab.cap.release()
        
        event.accept()
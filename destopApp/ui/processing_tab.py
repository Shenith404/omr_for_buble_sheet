import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QFileDialog, QProgressBar,
    QMessageBox, QSizePolicy, QApplication
)
from PySide6.QtCore import Qt, Signal, QThread

# Import the processing handlers
from .processing import (
    OMRProcessor,
    ImageDisplayHandler,
    ImageNavigationHandler,
    FileOperationHandler,
    ModelAnswersHandler,
    UIProcessingHandler,
    UIStateHandler
)


class ProcessingTab(QWidget):
    """
    Optimized Processing Tab for handling large batches of OMR sheets
    Features:
    - Memory-efficient image handling
    - Batch processing
    - Progress tracking
    - Error recovery
    """
    
    # Processing signals
    processing_complete = Signal(object, object, object)
    processing_cancelled = Signal()
    processing_started = Signal()
    processing_finished = Signal()
    
    def __init__(self):
        super().__init__()
        self.project_path = None
        self.image_paths = []
        self.current_index = 0
        self.processed_images = {}  # Stores only paths to processed images
        self.current_answers = {}  # Stores answers for CSV output
        self.processed_count = 0  # Track the number of processed images
        self.setup_ui()
        self.setup_connections()
        self.model_answers = []  # Placeholder for model answers
        self.handler=None
        
        # Initialize UI state
        self.image_label.setAlignment(Qt.AlignCenter)
        self.lbl_image_info.setText("0/0 images loaded")
        self.update_navigation_buttons()

    def setup_ui(self):
        """Initialize all UI components with optimized layouts"""
        # Apply dark theme
        self.setStyleSheet("""
            QWidget {
                background-color: #181818;
                color: #e8e8e8;
                font-family: 'Segoe UI', 'SF Pro Display', 'Inter', 'Roboto', sans-serif;
            }
            QGroupBox {
                background: #1e1e1e;
                border: 1px solid #404040;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 600;
                color: #ffffff;
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
            QPushButton {
                background: linear-gradient(180deg, #0078d4 0%, #005a9e 100%);
                border: 1px solid #004578;
                border-radius: 6px;
                color: #ffffff;
                font-size: 13px;
                font-weight: 500;
                padding: 10px 20px;
                min-width: 120px;
            }
            QPushButton:hover {
                background: linear-gradient(180deg, #1084d8 0%, #0066b2 100%);
                border-color: #0078d4;
            }
            QPushButton:pressed {
                background: linear-gradient(180deg, #005a9e 0%, #004578 100%);
            }
            QPushButton:disabled {
                background: #3a3a3a;
                border-color: #525252;
                color: #7a7a7a;
            }
            QLabel {
                color: #e8e8e8;
                font-weight: 400;
            }
            QProgressBar {
                border: 1px solid #404040;
                border-radius: 6px;
                text-align: center;
                font-size: 11px;
                font-weight: 500;
                background: #2a2a2a;
                color: #e8e8e8;
                height: 20px;
            }
            QProgressBar::chunk {
                background: linear-gradient(90deg, #0078d4 0%, #00bcf2 100%);
                border-radius: 5px;
                margin: 1px;
            }
        """)

        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(20)

        # Project Status Group
        project_group = QGroupBox("Project Status")
        project_layout = QHBoxLayout()
        project_layout.setSpacing(15)
        
        self.lbl_project = QLabel("No project loaded")
        self.lbl_project.setStyleSheet("""
            QLabel {
                background: #2a2a2a;
                border: 1px solid #404040;
                border-radius: 6px;
                padding: 10px 12px;
                color: #cccccc;
                font-style: italic;
            }
        """)
        
        project_layout.addWidget(self.lbl_project, 1)
        project_group.setLayout(project_layout)

        # Image Navigation Group
        nav_group = QGroupBox("Image Navigation")
        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(15)
        
        self.btn_prev = QPushButton("◀ Previous")
        self.btn_prev.setStyleSheet("""
            QPushButton {
                background: linear-gradient(180deg, #6c757d 0%, #495057 100%);
                border: 1px solid #495057;
                min-width: 100px;
            }
            QPushButton:hover {
                background: linear-gradient(180deg, #868e96 0%, #6c757d 100%);
            }
        """)
        
        self.btn_next = QPushButton("Next ▶")
        self.btn_next.setStyleSheet("""
            QPushButton {
                background: linear-gradient(180deg, #6c757d 0%, #495057 100%);
                border: 1px solid #495057;
                min-width: 100px;
            }
            QPushButton:hover {
                background: linear-gradient(180deg, #868e96 0%, #6c757d 100%);
            }
        """)
        
        self.btn_delete_image = QPushButton("Delete Image")
        self.btn_delete_image.setStyleSheet("""
            QPushButton {
                background: linear-gradient(180deg, #dc3545 0%, #c82333 100%);
                border: 1px solid #c82333;
                font-weight: 600;
                min-width: 120px;
            }
            QPushButton:hover {
                background: linear-gradient(180deg, #e55a67 0%, #dc3545 100%);
            }
        """)
        self.btn_delete_image.clicked.connect(self.delete_current_image)
        
        self.lbl_image_info = QLabel("0/0 images loaded")
        self.lbl_image_info.setStyleSheet("""
            QLabel {
                background: #2a2a2a;
                border: 1px solid #404040;
                border-radius: 6px;
                padding: 8px 12px;
                color: #ffffff;
                font-weight: 500;
                min-width: 150px;
            }
        """)
        self.lbl_image_info.setAlignment(Qt.AlignCenter)
        
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.lbl_image_info)
        nav_layout.addWidget(self.btn_next)
        nav_layout.addWidget(self.btn_delete_image)
        nav_group.setLayout(nav_layout)

        # Image Display with optimized settings
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2a2a2a, stop:1 #1e1e1e);
                border: 2px dashed #505050;
                border-radius: 8px;
                color: #888888;
                font-size: 16px;
                font-weight: 500;
                min-height: 400px;
            }
        """)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setText("📄 No images loaded\n\nOpen a project from the Project tab to begin processing")

        # Processing Controls with batch options
        # Processing Controls with batch options
        control_group = QGroupBox("Batch Processing")
        control_layout = QVBoxLayout()
        control_layout.setSpacing(12)
        
        # Create a horizontal layout for action buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)
        
        self.btn_save_model_answers = QPushButton("Save Answers")
        self.btn_save_model_answers.setStyleSheet("""
            QPushButton {
                background: linear-gradient(180deg, #6f42c1 0%, #5a2d8c 100%);
                border: 1px solid #5a2d8c;
                font-weight: 600;
                min-width: 120px;
            }
            QPushButton:hover {
                background: linear-gradient(180deg, #8b5cf6 0%, #6f42c1 100%);
            }
        """)
        self.btn_save_model_answers.setEnabled(False)

        self.btn_edit_answers = QPushButton("Edit Answers")
        self.btn_edit_answers.setStyleSheet("""
            QPushButton {
                background: linear-gradient(180deg, #fd7e14 0%, #e55a00 100%);
                border: 1px solid #e55a00;
                font-weight: 600;
                min-width: 120px;
            }
            QPushButton:hover {
                background: linear-gradient(180deg, #ff9500 0%, #fd7e14 100%);
            }
        """)
        self.btn_edit_answers.setEnabled(False)
        
        self.btn_process_all = QPushButton("Mark All Images")
        self.btn_process_all.setStyleSheet("""
            QPushButton {
                background: linear-gradient(180deg, #28a745 0%, #1e7e34 100%);
                border: 1px solid #1e7e34;
                font-weight: 600;
                min-width: 140px;
                font-size: 14px;
            }
            QPushButton:hover {
                background: linear-gradient(180deg, #34d058 0%, #28a745 100%);
            }
        """)
        
        self.btn_cancel = QPushButton("Cancel Processing")
        self.btn_cancel.setStyleSheet("""
            QPushButton {
                background: linear-gradient(180deg, #dc3545 0%, #c82333 100%);
                border: 1px solid #c82333;
                font-weight: 600;
                min-width: 140px;
            }
            QPushButton:hover {
                background: linear-gradient(180deg, #e55a67 0%, #dc3545 100%);
            }
        """)
        self.btn_cancel.setEnabled(False)
        
        # Add buttons to horizontal layout
        button_layout.addWidget(self.btn_save_model_answers)
        button_layout.addWidget(self.btn_edit_answers)
        button_layout.addStretch()
        button_layout.addWidget(self.btn_process_all)
        button_layout.addWidget(self.btn_cancel)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setValue(0)  # Set initial value
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #404040;
                border-radius: 10px;
                text-align: center;
                font-size: 13px;
                font-weight: 600;
                background-color: #2a2a2a;
                color: #ffffff;
                height: 28px;
                padding: 2px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #28a745, stop:0.5 #20c997, stop:1 #17a2b8);
                border-radius: 8px;
                margin: 1px;
                min-width: 10px;
            }
            QProgressBar[value="0"] {
                color: #888888;
            }
        """)
        
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet("""
            QLabel {
                background: #2a2a2a;
                border: 1px solid #404040;
                border-radius: 6px;
                padding: 10px 12px;
                color: #ffffff;
                font-weight: 500;
                font-size: 13px;
            }
        """)
        self.lbl_status.setAlignment(Qt.AlignCenter)

        control_layout.addLayout(button_layout)
        control_layout.addWidget(self.progress_bar)
        control_layout.addWidget(self.lbl_status)
        control_group.setLayout(control_layout)

        # Assemble main layout
        self.layout.addWidget(project_group)
        self.layout.addWidget(nav_group)
        self.layout.addWidget(self.image_label, 1)  # Give image label stretch
        self.layout.addWidget(control_group)
        self.setLayout(self.layout)

    def setup_connections(self):
        """Connect all signals and slots"""
        self.btn_prev.clicked.connect(self.show_previous_image)
        self.btn_next.clicked.connect(self.show_next_image)
        self.btn_process_all.clicked.connect(self.start_processing)
        self.btn_cancel.clicked.connect(self.cancel_processing)
        self.btn_save_model_answers.clicked.connect(self.save_model_answers)
        self.btn_edit_answers.clicked.connect(self.edit_answers)  # Connect the new button

    def test_progress_bar(self):
        """Test method to verify progress bar styling"""
        import time
        for i in range(0, 101, 10):
            self.progress_bar.setValue(i)
            self.lbl_status.setText(f"Testing progress: {i}%")
            QApplication.processEvents()
            time.sleep(0.1)

    def load_project(self, project_path):
        """Load project with optimized file handling"""
        self.project_path = project_path
        
        # Update UI state
        UIStateHandler.update_project_ui(
            project_path, 
            self.lbl_project, 
            self.btn_save_model_answers, 
            self.btn_edit_answers
        )
        
        # Reset state
        state = UIStateHandler.reset_processing_state()
        self.image_paths = []
        self.processed_images = state['processed_images']
        self.current_answers = state['current_answers']
        self.processed_count = state['processed_count']
        
        # Load images with error handling
        try:
            # Load model answers if available
            self.save_model_answers(True)
            # disable save answers button
            self.btn_save_model_answers.setEnabled(False)  # Disable the button after loading
            
            # Load images using handler
            self.image_paths = FileOperationHandler.load_project_images(project_path)
        
        except Exception as e:
            QMessageBox.warning(self, "Load Error", str(e))
        
        # Update UI
        if self.image_paths:
            self.current_index = 0
            self.show_current_image()
        
        UIStateHandler.update_image_info_ui(self.image_paths, self.lbl_image_info)
        self.update_navigation_buttons()
        self.btn_save_model_answers.setEnabled(True)  # Enable the button when a project is loaded
        self.btn_edit_answers.setEnabled(True)  # Enable the edit button when a project is loaded

    def show_current_image(self):
        """Display the current image without showing marked answers"""
        if not self.image_paths:
            self.image_label.clear()
            self.lbl_image_info.setText("No images loaded")
            return

        image_path = self.image_paths[self.current_index]
        ImageDisplayHandler.display_original_image(
            image_path, 
            self.image_label, 
            self.lbl_image_info, 
            self.current_index, 
            len(self.image_paths)
        )

    def show_next_image(self):
        """Navigate to the next image with bounds checking"""
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.show_current_image()
        self.update_navigation_buttons()

    def show_previous_image(self):
        """Navigate to the previous image with bounds checking"""
        if self.current_index > 0:
            self.current_index -= 1
            self.show_current_image()
        self.update_navigation_buttons()
            
    def update_navigation_buttons(self):
        """Update button states based on current position"""
        ImageNavigationHandler.update_navigation_buttons(
            self.btn_prev,
            self.btn_next,
            self.btn_process_all,
            self.image_paths,
            self.current_index
        )

    def start_processing(self):
        """Start optimized batch processing"""

        # give and exception when save answers button is enabled
        if self.btn_save_model_answers.isEnabled():
            QMessageBox.warning(self, "Error", "Please check & save model answers before processing.")
            return

        try:
            # Initialize processing using handler
            self.handler = UIProcessingHandler.initialize_processing(
                self.project_path, 
                self.image_paths, 
                self.model_answers
            )
            
            # Reset processing state
            state = UIStateHandler.reset_processing_state()
            self.processed_images = state['processed_images']
            self.current_answers = state['current_answers']
            self.processed_count = state['processed_count']
            
            # Setup UI for processing
            self.processing_started.emit()
            UIStateHandler.enable_processing_buttons(
                self.btn_process_all, 
                self.btn_cancel, 
                self.btn_delete_image, 
                enable_processing=False
            )
            self.progress_bar.setValue(0)
            self.lbl_status.setText("Initializing batch processing...")
            QApplication.processEvents()  # Ensure UI updates
            
            # Setup worker thread using handler
            self.worker_thread, self.omr_processor = UIProcessingHandler.setup_worker_thread(
                self.image_paths, 
                self.project_path, 
                self.model_answers,
                OMRProcessor
            )
            
            # Connect signals
            self.worker_thread.started.connect(self.omr_processor.process_all)
            self.omr_processor.progress_updated.connect(self.update_progress)
            self.omr_processor.image_processed.connect(
                lambda filename, answers, marked_image: self.save_image_results(filename, answers, marked_image, self.omr_processor.total_marks)
            )
            self.omr_processor.processing_complete.connect(self.finish_processing)
            self.omr_processor.error_occurred.connect(self.handle_processing_error)
            self.omr_processor.cancelled.connect(self.cancel_processing)
            
            # Cleanup connections
            self.omr_processor.processing_complete.connect(self.worker_thread.quit)
            self.omr_processor.error_occurred.connect(self.worker_thread.quit)
            self.worker_thread.finished.connect(self.worker_thread.deleteLater)
            
            # Start processing
            self.worker_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Processing Error", str(e))
            return

    def update_progress(self, progress, message):
        """Throttled progress updates"""
        UIProcessingHandler.update_progress_ui(
            progress, 
            message, 
            self.progress_bar, 
            self.lbl_status
        )


    def save_image_results(self, filename, answers, marked_image, total_marks):
        """Save results immediately after each image is processed"""
        try:
            # Save using handler
            FileOperationHandler.save_image_results(
                filename, 
                answers, 
                marked_image, 
                total_marks, 
                self.project_path, 
                self.handler
            )
            
            # Update UI
            self.display_marked_image(marked_image)
            self.processed_count += 1  # Increment processed count
            if self.current_index < len(self.image_paths) - 1:
                self.current_index += 1
            self.update_navigation_buttons()
            
        except Exception as e:
            self.handle_processing_error(str(e))

    def display_marked_image(self, marked_image):
        """Display marked image with optimized rendering"""
        ImageDisplayHandler.display_marked_image(
            marked_image,
            self.image_label,
            self.lbl_image_info,
            self.current_index,
            len(self.image_paths)
        )

    def finish_processing(self):
        """Verify all images were processed and allow navigation to other tabs"""
        UIProcessingHandler.finish_processing_ui(
            self.processed_count,
            len(self.image_paths),
            self.project_path,
            self.progress_bar,
            self.lbl_status,
            self.btn_process_all,
            self.btn_cancel,
            self.btn_delete_image,
            self.worker_thread
        )

        # Emit a signal or update the UI to allow navigation
        self.processing_finished.emit()

    def cancel_processing(self):
        """Handle processing cancellation and reset state"""
        UIProcessingHandler.cancel_processing_ui(
            self.worker_thread,
            getattr(self, 'omr_processor', None),
            self.progress_bar,
            self.btn_process_all,
            self.btn_cancel
        )

        # Reset processing state
        state = UIStateHandler.reset_processing_state()
        self.processed_images = state['processed_images']
        self.current_answers = state['current_answers']
        self.processed_count = state['processed_count']

        # Emit a signal or update the UI to allow navigation
        self.processing_cancelled.emit()

    def handle_processing_error(self, error_msg):
        """Handle processing errors with user feedback"""
        UIProcessingHandler.handle_processing_error_ui(
            error_msg,
            self.btn_process_all,
            self.btn_cancel,
            self.lbl_status,
            self.progress_bar
        )

    def resizeEvent(self, event):
        """Handle window resize to maintain image display"""
        super().resizeEvent(event)
        if hasattr(self, 'image_label') and self.image_paths:
            self.show_current_image()

    def set_image_paths(self, image_paths):
        """
        Set the image paths to process
        Args:
            image_paths: List of paths to images
        """
        # Validate input using handler
        ImageNavigationHandler.validate_image_paths(image_paths)
        
        # Store paths and reset state
        self.image_paths = [os.path.normpath(str(p)) for p in image_paths]  # Normalize paths
        self.current_index = 0
        
        # Reset processing state
        state = UIStateHandler.reset_processing_state()
        self.processed_images = state['processed_images']
        
        # Update UI
        if self.image_paths:
            self.show_current_image()
        else:
            self.image_label.clear()
            self.lbl_image_info.setText("No images loaded")
        
        self.update_navigation_buttons()
        self.btn_process_all.setEnabled(len(self.image_paths) > 0)

    def closeEvent(self, event):
        """Clean up resources when closing"""
        if hasattr(self, 'worker_thread') and self.worker_thread.isRunning():
            self.cancel_processing()
            self.worker_thread.quit()
            self.worker_thread.wait(1000)  # Wait up to 1 second
        event.accept()
        self.btn_save_model_answers.setEnabled(False)  # Disable the button when the application is closed

    def showEvent(self, event):
        """Reload images every time the tab is shown"""
        super().showEvent(event)
        if self.project_path:
            self.load_project(self.project_path)

    def delete_current_image(self):
        """Delete the currently previewed image permanently from the project"""
        if not self.image_paths:
            QMessageBox.warning(self, "Delete Error", "No images to delete.")
            return

        try:
            # Get the current image path
            image_path = self.image_paths[self.current_index]
            filename = os.path.basename(image_path)

            # Confirm deletion
            reply = QMessageBox.question(
                self,
                "Delete Image",
                f"Are you sure you want to delete '{filename}' permanently?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

            # Delete using handler
            success_message = FileOperationHandler.delete_image_file(image_path)

            # Remove the image from the list
            del self.image_paths[self.current_index]

            # Update the current index
            if self.current_index >= len(self.image_paths):
                self.current_index = max(0, len(self.image_paths) - 1)

            # Update the UI
            if self.image_paths:
                self.show_current_image()
            else:
                self.image_label.clear()
                self.lbl_image_info.setText("No images loaded")

            self.update_navigation_buttons()
            QMessageBox.information(self, "Delete Successful", success_message)

        except Exception as e:
            QMessageBox.critical(self, "Delete Error", str(e))

    def save_model_answers(self,isInitialSave=False):
        """Load or create model answers XLSX file and open it for editing"""
        try:
            self.model_answers = ModelAnswersHandler.save_model_answers_workflow(self.project_path)
            if not isInitialSave:
                QMessageBox.information(self, "Success", "Model answers saved successfully!")
            # Disable save answers button
            self.btn_save_model_answers.setEnabled(False)

        except Exception as e:
            QMessageBox.warning(self, "Error", f"An error occurred: {str(e)}. Recreating the file.")
            try:
                model_answers_path = os.path.join(self.project_path, "model_answers.xlsx")
                if os.path.exists(model_answers_path):
                    os.remove(model_answers_path)
                ModelAnswersHandler.create_model_answers_file(model_answers_path)
                ModelAnswersHandler.open_file_for_editing(model_answers_path)
                QMessageBox.information(self, "Notice", "Please add the answers again.")
            except Exception as recreate_error:
                QMessageBox.critical(self, "Critical Error", f"Failed to recreate the file: {str(recreate_error)}")

    def edit_answers(self):
        """Open the model answers XLSX file for editing"""
        try:
            ModelAnswersHandler.edit_answers_workflow(self.project_path)
            self.btn_save_model_answers.setEnabled(True)  # Enable save button when editing
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

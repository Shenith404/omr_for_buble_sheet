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
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(10, 10, 10, 10)

        # Project Selection Group
        project_group = QGroupBox("Project Selection")
        project_layout = QHBoxLayout()
        self.btn_select_project = QPushButton("Select Project")
        self.btn_select_project.setStyleSheet("font-weight: bold;")
        self.lbl_project = QLabel("No project selected")
        project_layout.addWidget(self.btn_select_project)
        project_layout.addWidget(self.lbl_project)
        project_group.setLayout(project_layout)

        # Image Navigation Group
        nav_group = QGroupBox("Image Navigation")
        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("◀ Previous")
        self.btn_next = QPushButton("Next ▶")
        self.btn_delete_image = QPushButton("Delete Image")  # Move delete button here
        self.btn_delete_image.setStyleSheet("""
            background-color: #f44336; 
            color: white;
            font-weight: bold;
        """)  # Add red background
        self.btn_delete_image.clicked.connect(self.delete_current_image)
        self.lbl_image_info = QLabel("0/0 images loaded")
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.lbl_image_info)
        nav_layout.addWidget(self.btn_next)
        nav_layout.addWidget(self.btn_delete_image)  # Add delete button to navigation layout
        nav_group.setLayout(nav_layout)

        # Image Display with optimized settings
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            border: 1px solid gray; 
            min-height: 400px;
            background-color: #f0f0f0;
        """)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Processing Controls with batch options
        control_group = QGroupBox("Batch Processing")
        control_layout = QVBoxLayout()
        self.btn_process_all = QPushButton("Mark All Images")
        self.btn_process_all.setStyleSheet("""
            background-color: #4CAF50; 
            color: white;
            padding: 8px;
            font-weight: bold;
        """)
        self.btn_cancel = QPushButton("Cancel Processing")
        self.btn_cancel.setStyleSheet("""
            background-color: #f44336;
            color: white;
            padding: 8px;
        """)
        self.btn_cancel.setEnabled(False)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet("font-weight: bold;")
        
        # Add a button to load model answers
        self.btn_save_model_answers = QPushButton("Save Answers")
        self.btn_save_model_answers.setStyleSheet("font-weight: bold; " )
        self.btn_save_model_answers.setEnabled(False)  # Initially disabled

        # Add new button for editing answers
        self.btn_edit_answers = QPushButton("Edit Answers")
        self.btn_edit_answers.setStyleSheet("font-weight: bold;")
        self.btn_edit_answers.setEnabled(False)  # Initially disabled

        control_layout.addWidget(self.btn_save_model_answers)  # Add to the control group
        control_layout.addWidget(self.btn_edit_answers)  # Add the new button
        control_layout.addWidget(self.btn_process_all)
        control_layout.addWidget(self.btn_cancel)
        control_layout.addWidget(self.progress_bar)
        control_layout.addWidget(self.lbl_status)
        control_group.setLayout(control_layout)

        # Assemble main layout
        self.layout.addWidget(project_group)
        self.layout.addWidget(nav_group)
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(control_group)
        self.setLayout(self.layout)

    def setup_connections(self):
        """Connect all signals and slots"""
        self.btn_select_project.clicked.connect(self.select_project)
        self.btn_prev.clicked.connect(self.show_previous_image)
        self.btn_next.clicked.connect(self.show_next_image)
        self.btn_process_all.clicked.connect(self.start_processing)
        self.btn_cancel.clicked.connect(self.cancel_processing)
        self.btn_save_model_answers.clicked.connect(self.save_model_answers)
        self.btn_edit_answers.clicked.connect(self.edit_answers)  # Connect the new button

    def select_project(self):
        """Let user select a project folder with validation"""
        project_path = QFileDialog.getExistingDirectory(
            self, 
            "Select Project Folder",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        if project_path:
            self.load_project(project_path)

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
            self.save_model_answers()
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

    def save_model_answers(self):
        """Load or create model answers XLSX file and open it for editing"""
        try:
            self.model_answers = ModelAnswersHandler.save_model_answers_workflow(self.project_path)
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
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

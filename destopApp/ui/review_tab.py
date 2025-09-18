import os
import json
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QSizePolicy, QFileDialog,
    QComboBox, QSpacerItem, QMessageBox, QSpinBox,
    QGridLayout, QScrollArea, QFrame, QToolTip,
    QLineEdit, QListWidget, QListWidgetItem
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt, Signal
import db  
import utils
import cv2
import shutil

# Import the extracted modules
from .review_handlers import (
    ReviewUIBuilder,
    ReviewImageHandler,
    ReviewFileHandler,
    ReviewOMRHandler
)

class ReviewTab(QWidget):
    """Enhanced Review Tab with project loading and 70-30 split layout"""
    
    # Signals
    project_loaded = Signal(str)
    navigation_requested = Signal(int)  # Request to switch to Processing tab with index
    answer_modified = Signal(str, int, int)  # filename, question_num, new_answer
    
    def __init__(self):
        super().__init__()
        self.project_path = None
        self.image_paths = []
        self.filtered_image_paths = []  # Store filtered results
        self.original_image_paths = []  # Store original images before processing
        self.current_index = 0
        self.answers = {}
        self.reviewed_images = set()
        self.setup_ui()
        self.setup_connections()
        self.handler = None
        self.model_answers = []
        self.selected_question = 1  # Track currently selected question
        self.question_buttons = []  # Store question button references
    
        
    def setup_ui(self):
        """Initialize UI with 70-30 horizontal split layout"""
        # Apply professional dark theme
        self.setStyleSheet(ReviewUIBuilder.get_main_stylesheet())

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Left Panel (70%)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(15)
        
        # Project Status Group
        project_group, self.lbl_project = ReviewUIBuilder.create_project_status_group()
        
        # Image Navigation Group
        nav_group = QGroupBox("Image Navigation")
        nav_layout = QVBoxLayout()
        nav_layout.setSpacing(15)
        
        # Search and Filter Row
        search_layout = QHBoxLayout()
        search_layout.setSpacing(10)
        
        # Search input and filter combo
        self.search_input, self.filter_combo = ReviewUIBuilder.create_search_and_filter_widgets()
        search_layout.addWidget(self.search_input, 2)
        search_layout.addWidget(self.filter_combo, 1)
        
        # Navigation buttons row
        nav_buttons_layout = QHBoxLayout()
        nav_buttons_layout.setSpacing(15)
        
        self.btn_prev, self.btn_next, self.lbl_image_info = ReviewUIBuilder.create_navigation_buttons()
        nav_buttons_layout.addWidget(self.btn_prev)
        nav_buttons_layout.addWidget(self.lbl_image_info)
        nav_buttons_layout.addWidget(self.btn_next)
        
        nav_layout.addLayout(search_layout)
        nav_layout.addLayout(nav_buttons_layout)
        nav_group.setLayout(nav_layout)
        
        # Image Display
        self.image_label = ReviewUIBuilder.create_image_display_label()
        
        # Results Info Group
        info_group, self.lbl_results_info = ReviewUIBuilder.create_results_info_group()
        
        # Assemble left panel (70%)
        left_layout.addWidget(project_group)
        left_layout.addWidget(nav_group)
        left_layout.addWidget(self.image_label, 1)  # Expand image area
        left_layout.addWidget(info_group)
        
        # Right Panel (30%) - Now with all requested controls
        right_panel = QWidget()
        right_panel.setStyleSheet("""
            QWidget {
                border-left: 2px solid #404040;
                padding-left: 15px;
            }
        """)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(15, 0, 0, 0)
        right_layout.setSpacing(20)
        
        # Download Results Button (added at top of right panel)
        self.btn_download_results = ReviewUIBuilder.create_download_results_button()
        right_layout.addWidget(self.btn_download_results)
        
        # Image List Group
        images_group = QGroupBox("Images")
        images_layout = QVBoxLayout(images_group)
        images_layout.setSpacing(10)
        
        self.images_list = ReviewUIBuilder.create_images_list_widget()
        images_layout.addWidget(self.images_list)
        right_layout.addWidget(images_group)
        
        # File Rename Section
        rename_group = QGroupBox("Rename Image File")
        rename_layout = QVBoxLayout(rename_group)
        rename_layout.setSpacing(8)
        rename_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create rename widgets
        (self.current_filename_label, self.filename_input, 
         self.file_extension_label, self.btn_rename_file) = ReviewUIBuilder.create_file_rename_widgets()
        
        # New filename input layout
        rename_input_layout = QHBoxLayout()
        rename_input_layout.setSpacing(5)
        rename_input_layout.addWidget(self.filename_input, 1)
        rename_input_layout.addWidget(self.file_extension_label)
        rename_input_layout.addWidget(self.btn_rename_file)
        
        rename_layout.addWidget(self.current_filename_label)
        rename_layout.addLayout(rename_input_layout)
        right_layout.addWidget(rename_group)
        
        # Question Selection Group (Enhanced UI)
        question_group = QGroupBox("Change Detected Answers")
        question_layout = QVBoxLayout(question_group)
        question_layout.setSpacing(12)
        
        # Question Selection Info
        selection_info = QLabel("Click a question number to select:")
        selection_info.setStyleSheet("QLabel { color: #b8b8b8; font-size: 12px; margin-bottom: 5px; }")
        question_layout.addWidget(selection_info)
        
        # Initialize question buttons list
        self.question_buttons = []
        
        # Question Grid Container with Scroll
        scroll_area = ReviewUIBuilder.create_question_grid(self.question_buttons)
        question_layout.addWidget(scroll_area)
        
        # Set first question as selected by default
        if self.question_buttons:
            self.question_buttons[0].setChecked(True)
        
        # Selected Question Info
        self.selected_question_label = ReviewUIBuilder.create_selected_question_label()
        question_layout.addWidget(self.selected_question_label)
        
        # Answer Section (Enhanced)
        answer_label = QLabel("New Answer:")
        answer_label.setStyleSheet("QLabel { color: #e8e8e8; font-weight: 500; margin-bottom: 3px; }")
        
        # Answer buttons and layout
        answer_buttons_layout = QHBoxLayout()
        answer_buttons_layout.setSpacing(6)
        
        self.answer_buttons = ReviewUIBuilder.create_answer_buttons()
        for i, btn in enumerate(self.answer_buttons):
            btn.clicked.connect(lambda checked, ans=i+1: self.select_answer(ans))
            answer_buttons_layout.addWidget(btn)
        
        # Set first answer as selected by default
        if self.answer_buttons:
            self.answer_buttons[0].setChecked(True)
        
        # Legacy combo box (hidden but kept for compatibility)
        self.answer_combo = QComboBox()
        self.answer_combo.addItems([str(i) for i in range(1, 5)])
        self.answer_combo.hide()  # Hide the old combo box
        
        # Change Answer and Mark as Reviewed buttons
        self.btn_change_answer, self.btn_mark_reviewed = ReviewUIBuilder.create_action_buttons()
        
        question_layout.addWidget(answer_label)
        question_layout.addLayout(answer_buttons_layout)
        question_layout.addWidget(self.btn_change_answer)
        
        # Add question group to right layout
        right_layout.addWidget(question_group)
        right_layout.addStretch(1)  # Add flexible space
        
        # Mark as Reviewed button at bottom
        right_layout.addWidget(self.btn_mark_reviewed)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel, 7)  # 70% width
        main_layout.addWidget(right_panel, 3)  # 30% width
        
        # Connect question button signals
        for i, btn in enumerate(self.question_buttons):
            btn.clicked.connect(lambda checked, q=i+1: self.select_question(q))
        
        self.update_navigation_buttons()
        self.update_review_button_state()
        
        # Enable focus for keyboard shortcuts
        self.setFocusPolicy(Qt.StrongFocus)
        

    def setup_connections(self):
        """Connect all signals and slots"""
        self.btn_prev.clicked.connect(self.show_previous_image)
        self.btn_next.clicked.connect(self.show_next_image)
        self.btn_mark_reviewed.clicked.connect(self.mark_as_reviewed)
        self.btn_change_answer.clicked.connect(self.change_detected_answer)
        self.btn_download_results.clicked.connect(self.download_results)
        
        # Connect search and filter functionality
        self.search_input.textChanged.connect(self.apply_filters)
        self.filter_combo.currentTextChanged.connect(self.apply_filters)
        self.images_list.itemClicked.connect(self.on_image_selected)
        
        # Connect file rename functionality
        self.btn_rename_file.clicked.connect(self.rename_current_file)
        self.filename_input.returnPressed.connect(self.rename_current_file)

    def select_question(self, question_num):
        """Handle question selection from grid"""
        self.selected_question = question_num
        
        # Update question selection using handler
        ReviewOMRHandler.update_question_selection(
            self.question_buttons, self.selected_question_label, question_num
        )
        
        # Update question button appearance with current answer if available
        self.update_question_button_appearance(question_num)
    
    def select_answer(self, answer_num):
        """Handle answer selection from buttons"""
        # Update answer selection using handler
        ReviewOMRHandler.update_answer_selection(
            self.answer_buttons, self.answer_combo, answer_num
        )
    
    def update_question_button_appearance(self, question_num=None):
        """Update question button appearance to show current answers"""
        if not self.image_paths or self.current_index >= len(self.image_paths):
            return

        current_image = os.path.basename(self.image_paths[self.current_index])
        
        # Update question button appearances using handler
        ReviewOMRHandler.update_question_button_appearance(
            self.question_buttons, self.handler, current_image
        )

    def apply_filters(self):
        """Apply search and review status filters to image list"""
        if not self.image_paths:
            self.filtered_image_paths = []
            self.update_image_list()
            return
            
        search_text = self.search_input.text()
        filter_type = self.filter_combo.currentText()
        
        # Start with all images
        filtered_paths = self.image_paths.copy()
        
        # Apply search filter
        filtered_paths = ReviewImageHandler.apply_search_filter(filtered_paths, search_text)
        
        # Apply review status filter
        filtered_paths = ReviewImageHandler.apply_review_status_filter(
            filtered_paths, filter_type, self.is_image_reviewed
        )
        
        self.filtered_image_paths = filtered_paths
        self.update_image_list()
        
        # Update current index if current image is not in filtered results
        if self.image_paths and self.current_index < len(self.image_paths):
            current_image_path = self.image_paths[self.current_index]
            if current_image_path not in self.filtered_image_paths:
                # Reset to first filtered image if available
                if self.filtered_image_paths:
                    self.current_index = self.image_paths.index(self.filtered_image_paths[0])
                    self.show_current_image()
    
    def is_image_reviewed(self, filename):
        """Check if image is reviewed using database"""
        return ReviewOMRHandler.is_image_reviewed(self.handler, filename, self.reviewed_images)
    
    def update_image_list(self):
        """Update the image list widget with filtered results"""
        ReviewImageHandler.update_image_list_widget(
            self.images_list, self.filtered_image_paths, self.is_image_reviewed
        )
        
        # Update image count info
        ReviewImageHandler.update_image_count_info(
            self.lbl_image_info, self.image_paths, self.filtered_image_paths, self.current_index
        )
    
    def on_image_selected(self, item):
        """Handle image selection from the list"""
        new_index = ReviewImageHandler.get_image_index_from_list_item(item, self.image_paths)
        if new_index >= 0:
            self.current_index = new_index
            self.show_current_image()
            self.update_navigation_buttons()
            self.update_image_list()  # Update to highlight current selection

    def load_project(self, project_path):
        """Load processed images from project's results folder"""
        self.project_path = project_path
        self.lbl_project.setText(f"Reviewing: {os.path.basename(project_path)}")
        self.image_paths = []
        self.filtered_image_paths = []
        self.answers = {}
        self.reviewed_images = set()
        
        # Load processed images from results folder
        self.image_paths = ReviewImageHandler.load_project_images(project_path)

        # Load db
        self.handler = db.OMRJsonHandler(project_path)
        # Load model answers
        self.model_answers = ReviewOMRHandler.load_model_answers(self.handler)
        
        # Initialize filtered paths and apply current filters
        self.filtered_image_paths = self.image_paths.copy()
        self.apply_filters()
        
        if self.image_paths:
            self.current_index = 0
            self.show_current_image()
        else:
            self.lbl_image_info.setText("No processed images found")
            ReviewImageHandler.show_no_images_message(self.image_label, "no_results")
        
        self.update_navigation_buttons()
        self.update_review_button_state()
        self.project_loaded.emit(project_path)

    def load_images(self, image_paths):
        """
        Prepare the review tab with original image paths.
        This method is called when images are added to the project but before processing.
        The review tab will show a message that processing is needed first.
        """
        # Store the original image paths for reference
        self.original_image_paths = image_paths.copy() if image_paths else []
        
        # Clear current state since these are not processed images yet
        self.image_paths = []
        self.filtered_image_paths = []
        self.answers = {}
        self.reviewed_images = set()
        self.current_index = 0
        
        # Clear the image list
        self.images_list.clear()
        
        # Show message that processing is needed
        if self.original_image_paths:
            self.lbl_image_info.setText(f"{len(self.original_image_paths)} images added - Processing required")
            ReviewImageHandler.show_no_images_message(self.image_label, "processing_required")
        else:
            self.lbl_image_info.setText("No images loaded")
            ReviewImageHandler.show_no_images_message(self.image_label, "no_images")
        
        # Disable navigation since there are no processed images yet
        self.update_navigation_buttons()
        self.update_review_button_state()

    def show_current_image(self):
        """Display the current processed image with results"""
        if not self.image_paths:
            return
            
        image_path = self.image_paths[self.current_index]
        filename = os.path.basename(image_path)
        
        try:
            # Load and display image
            success = ReviewImageHandler.display_current_image(image_path, self.image_label)
            
            if success:
                # Update image info with filtered count
                self.update_image_list()
                
                # Highlight current image in the list
                ReviewImageHandler.highlight_current_image_in_list(self.images_list, image_path)
                
                # Update review button state
                self.update_review_button_state()
                
                # Update question button appearances for this image
                self.update_question_button_appearance()
                
                # Update filename display for rename section
                self.update_filename_display()
                
        except Exception as e:
            ReviewImageHandler.show_image_loading_error(self.image_label, str(e))
            self.lbl_results_info.setText("")

    def show_next_image(self):
        """Navigate to the next image"""
        if ReviewImageHandler.validate_image_navigation(self.current_index, len(self.image_paths), "next"):
            self.current_index += 1
            self.show_current_image()
        self.update_navigation_buttons()

    def show_previous_image(self):
        """Navigate to the previous image"""
        if ReviewImageHandler.validate_image_navigation(self.current_index, len(self.image_paths), "prev"):
            self.current_index -= 1
            self.show_current_image()
        self.update_navigation_buttons()

    def update_navigation_buttons(self):
        """Update button states based on current position"""
        ReviewImageHandler.update_navigation_button_states(
            self.btn_prev, self.btn_next, self.current_index, len(self.image_paths)
        )
        self.btn_change_answer.setEnabled(len(self.image_paths) > 0)
        self.btn_download_results.setEnabled(len(self.image_paths) > 0)

    def update_review_button_state(self):
        """Update the Mark as Reviewed button state"""
        if not self.image_paths:
            self.btn_mark_reviewed.setEnabled(False)
            return
            
        current_image = os.path.basename(self.image_paths[self.current_index])
        ReviewOMRHandler.update_review_button_state(
            self.btn_mark_reviewed, self.handler, current_image, self.reviewed_images
        )

    def mark_as_reviewed(self):
        """Mark current image as reviewed"""
        if not self.image_paths:
            return
            
        current_image = os.path.basename(self.image_paths[self.current_index])
        
        try:
            updated_image_path = ReviewOMRHandler.mark_as_reviewed(
                self.handler, self.project_path, current_image, self.reviewed_images
            )
            
            # Update the displayed image
            self.image_paths[self.current_index] = updated_image_path
            self.show_current_image()
            
            # Update review button state and refresh filters
            self.update_review_button_state()
            self.apply_filters()  # Refresh the filtered list to reflect new status
            
        except Exception as e:
            print("Error marking as reviewed:", e)
            QMessageBox.critical(
                self, "Error", f"Failed to mark as reviewed: {str(e)}", QMessageBox.Ok
            )

    def change_detected_answer(self):
        """Change the detected answer for selected question"""
        if not self.image_paths:
            return
        
        # Disable button to prevent multiple clicks
        self.btn_change_answer.setEnabled(False)

        current_image = os.path.basename(self.image_paths[self.current_index])
        question_num = self.selected_question  # Use selected question from grid
        new_answer = int(self.answer_combo.currentText())
        
        try:
            # Validate input
            is_valid, error_msg = ReviewOMRHandler.validate_question_answer_input(question_num, new_answer)
            if not is_valid:
                QMessageBox.warning(self, "Invalid Input", error_msg)
                return
            
            # Update the answer using handler
            result_image_path = ReviewOMRHandler.change_detected_answer(
                self.handler, self.project_path, current_image, 
                question_num, new_answer, self.model_answers
            )
            
            # Update the displayed image
            self.image_paths[self.current_index] = result_image_path
            self.show_current_image()
            
            # Refresh the image list to maintain current state
            self.apply_filters()
            
            # Update question button appearance to reflect the change
            self.update_question_button_appearance()

            # Show confirmation
            QMessageBox.information(
                self,
                "Answer Updated",
                f"Question {question_num} answer changed to {new_answer} for {current_image}",
                QMessageBox.Ok
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Updating Answer",
                f"Failed to update answer: {str(e)}",
                QMessageBox.Ok
            )
        finally:
            # Enable button after processing
            self.btn_change_answer.setEnabled(True)

    def rename_current_file(self):
        """Rename the current image file"""
        if not self.image_paths or self.current_index >= len(self.image_paths):
            QMessageBox.warning(self, "No File Selected", "Please select an image first")
            return
        
        new_name = self.filename_input.text().strip()
        if not new_name:
            QMessageBox.warning(self, "Invalid Name", "Please enter a valid filename")
            return
        
        # Validate filename
        is_valid, error_msg = ReviewFileHandler.validate_filename(new_name)
        if not is_valid:
            QMessageBox.warning(self, "Invalid Name", error_msg)
            return
        
        current_path = self.image_paths[self.current_index]
        current_filename = os.path.basename(current_path)
        
        try:
            # Disable button during processing
            self.btn_rename_file.setEnabled(False)
            
            # Rename files using handler
            new_filename_with_ext, renamed_paths = ReviewFileHandler.rename_image_file(
                self.project_path, current_filename, new_name
            )
            
            # Update database entry if handler is available
            ReviewFileHandler.update_database_filename(self.handler, current_filename, new_filename_with_ext)
            
            # Update the current image paths
            new_path = os.path.join(self.project_path, "results", new_filename_with_ext)
            self.image_paths[self.current_index] = new_path
            
            # Update original image paths if they exist
            self.original_image_paths = ReviewFileHandler.update_image_paths_after_rename(
                self.original_image_paths, current_filename, new_filename_with_ext
            )
            
            # Clear the input field
            self.filename_input.clear()
            
            # Refresh displays
            self.update_filename_display()
            self.apply_filters()  # Refresh the image list
            
            QMessageBox.information(
                self,
                "File Renamed",
                f"File renamed from '{current_filename}' to '{new_filename_with_ext}'"
            )
            
        except FileExistsError as e:
            QMessageBox.warning(self, "File Exists", str(e))
        except Exception as e:
            QMessageBox.critical(self, "Rename Failed", f"Failed to rename file: {str(e)}")
        finally:
            self.btn_rename_file.setEnabled(True)

    def update_filename_display(self):
        """Update the current filename display"""
        if self.image_paths and self.current_index < len(self.image_paths):
            current_filename = os.path.basename(self.image_paths[self.current_index])
            name, ext = os.path.splitext(current_filename)
            
            self.current_filename_label.setText(f"Current: {current_filename}")
            self.file_extension_label.setText(ext)
            self.filename_input.setPlaceholderText(f"New name (currently: {name})")
        else:
            self.current_filename_label.setText("Current: No file selected")
            self.file_extension_label.setText(".jpg")
            self.filename_input.setPlaceholderText("Enter new filename...")

    def download_results(self):
        """Handle downloading of results to selected directory"""
        if not self.project_path:
            QMessageBox.warning(self, "No Project", "Please load a project first")
            return
        
        try:
            # Download results using handler
            dest_dir = ReviewFileHandler.download_project_results(self.project_path)
            
            if dest_dir:  # User didn't cancel
                # Create Excel sheet
                p_title = self.project_path.split(os.sep)[-1]
                excel_path = os.path.join(dest_dir, p_title + "_result.xlsx")
                ReviewFileHandler.export_results_to_excel(self.handler, excel_path)

                QMessageBox.information(
                    self,
                    "Download Complete",
                    f"All results downloaded successfully to:\n{dest_dir}",
                    QMessageBox.Ok
                )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Download Failed",
                f"Error downloading results: {str(e)}",
                QMessageBox.Ok
            )

    def resizeEvent(self, event):
        """Handle window resize to maintain image display"""
        super().resizeEvent(event)
        if self.image_paths:
            self.show_current_image()
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for better user experience"""
        key = event.key()
        
        # Question navigation with arrow keys
        if key == Qt.Key_Up or key == Qt.Key_Left:
            if self.selected_question > 1:
                self.select_question(self.selected_question - 1)
        elif key == Qt.Key_Down or key == Qt.Key_Right:
            if self.selected_question < 50:
                self.select_question(self.selected_question + 1)
        
        # Quick answer selection with number keys
        elif key == Qt.Key_1:
            self.select_answer(1)
        elif key == Qt.Key_2:
            self.select_answer(2)
        elif key == Qt.Key_3:
            self.select_answer(3)
        elif key == Qt.Key_4:
            self.select_answer(4)
        
        # Apply answer change with Enter
        elif key == Qt.Key_Return or key == Qt.Key_Enter:
            if self.btn_change_answer.isEnabled():
                self.change_detected_answer()
        
        # Image navigation
        elif key == Qt.Key_PageUp:
            self.show_previous_image()
        elif key == Qt.Key_PageDown:
            self.show_next_image()
        
        # Mark as reviewed with Space
        elif key == Qt.Key_Space:
            if self.btn_mark_reviewed.isEnabled():
                self.mark_as_reviewed()
        
        # Question quick jump with Ctrl+number (1-9 for questions 1-9)
        elif event.modifiers() == Qt.ControlModifier:
            if Qt.Key_1 <= key <= Qt.Key_9:
                question_num = key - Qt.Key_0  # Convert key to number
                if 1 <= question_num <= 50:
                    self.select_question(question_num)
        
        else:
            super().keyPressEvent(event)
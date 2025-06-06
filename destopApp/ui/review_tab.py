import os
import json
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QSizePolicy, QFileDialog,
    QComboBox, QSpacerItem, QMessageBox
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt, Signal
import db  # Assuming db is a module for handling JSON data
import utils
import cv2
import shutil

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
        self.current_index = 0
        self.answers = {}
        self.reviewed_images = set()
        self.setup_ui()
        self.setup_connections()
        self.handler = None
        self.model_answers = []
        
    def setup_ui(self):
        """Initialize UI with 70-30 horizontal split layout"""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Left Panel (70%)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        # Project Selection Group
        project_group = QGroupBox("Project Selection")
        project_layout = QHBoxLayout()
        
        self.btn_select_project = QPushButton("Open Project")
        self.btn_select_project.setStyleSheet("font-weight: bold;")
        self.lbl_project = QLabel("No project loaded")
        self.lbl_project.setStyleSheet("font-weight: bold;")
        
        project_layout.addWidget(self.btn_select_project)
        project_layout.addWidget(self.lbl_project)
        project_group.setLayout(project_layout)
        
        # Image Navigation Group
        nav_group = QGroupBox("Image Navigation")
        nav_layout = QHBoxLayout()
        
        self.btn_prev = QPushButton("◀ Previous")
        self.btn_next = QPushButton("Next ▶")
        self.lbl_image_info = QLabel("0/0 images loaded")
        
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.lbl_image_info)
        nav_layout.addWidget(self.btn_next)
        nav_group.setLayout(nav_layout)
        
        # Image Display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            border: 1px solid gray; 
            min-height: 400px;
            background-color: #f0f0f0;
        """)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Results Info Group
        info_group = QGroupBox("Results Information")
        info_layout = QVBoxLayout()
        
        self.lbl_results_info = QLabel("No results available")
        self.lbl_results_info.setWordWrap(True)
        info_layout.addWidget(self.lbl_results_info)
        
        info_group.setLayout(info_layout)
        
        # Assemble left panel (70%)
        left_layout.addWidget(project_group)
        left_layout.addWidget(nav_group)
        left_layout.addWidget(self.image_label, 1)  # Expand image area
        left_layout.addWidget(info_group)
        
        # Right Panel (30%) - Now with all requested controls
        right_panel = QWidget()
        right_panel.setStyleSheet("""
            border-left: 1px solid #ddd;
        """)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(15)
        
        # Download Results Button (added at top of right panel)
        self.btn_download_results = QPushButton("Download Results")
        self.btn_download_results.setStyleSheet("""
            QPushButton {
                font-weight: bold;
                padding: 10px;
                background-color: #2E7D32;
                color: white;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1B5E20;
            }
            QPushButton:disabled {
                background-color: #a5d6a7;
                color: #e8f5e9;
            }
        """)
        right_layout.addWidget(self.btn_download_results)
        
        # Question Selection Group
        question_group = QGroupBox("Change Detected Answers")
        question_layout = QVBoxLayout(question_group)
        question_layout.setSpacing(10)
        
        # First dropdown (1-50)
        self.question_combo = QComboBox()
        self.question_combo.addItems([str(i) for i in range(1, 51)])
        question_layout.addWidget(QLabel("Question Number:"))
        question_layout.addWidget(self.question_combo)
        
        # Second dropdown (1-4)
        self.answer_combo = QComboBox()
        self.answer_combo.addItems([str(i) for i in range(1, 5)])
        question_layout.addWidget(QLabel("Correct Answer:"))
        question_layout.addWidget(self.answer_combo)
        
        # Change Answer button
        self.btn_change_answer = QPushButton("Update Answer")
        self.btn_change_answer.setStyleSheet("""
            QPushButton {
                font-weight: bold;
                padding: 8px;
                background-color: #2196F3;
                color: white;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        question_layout.addWidget(self.btn_change_answer)
        
        # Add spacer
        right_layout.addWidget(question_group)
        right_layout.addStretch(1)
        
        # Mark as Reviewed button at bottom
        self.btn_mark_reviewed = QPushButton("✓ Mark as Reviewed")
        self.btn_mark_reviewed.setStyleSheet("""
            QPushButton {
                font-weight: bold;
                padding: 10px;
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #a5d6a7;
                color: #e8f5e9;
            }
        """)
        right_layout.addWidget(self.btn_mark_reviewed)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel, 7)  # 70% width
        main_layout.addWidget(right_panel, 3)  # 30% width
        
        self.update_navigation_buttons()
        self.update_review_button_state()

    def setup_connections(self):
        """Connect all signals and slots"""
        self.btn_select_project.clicked.connect(self.select_project)
        self.btn_prev.clicked.connect(self.show_previous_image)
        self.btn_next.clicked.connect(self.show_next_image)
        self.btn_mark_reviewed.clicked.connect(self.mark_as_reviewed)
        self.btn_change_answer.clicked.connect(self.change_detected_answer)
        self.btn_download_results.clicked.connect(self.download_results)

    def select_project(self):
        """Let user select a project folder"""
        project_path = QFileDialog.getExistingDirectory(
            self, 
            "Select Project Folder",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        if project_path:
            self.load_project(project_path)

    def load_project(self, project_path):
        """Load processed images from project's results folder"""
        self.project_path = project_path
        self.lbl_project.setText(f"Reviewing: {os.path.basename(project_path)}")
        self.image_paths = []
        self.answers = {}
        self.reviewed_images = set()
        
        # Load processed images from results folder
        results_dir = os.path.join(project_path, "results")
        if os.path.exists(results_dir):
            self.image_paths = [
                os.path.join(results_dir, f)
                for f in sorted(os.listdir(results_dir))
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ]

        # Load db
        self.handler = db.OMRJsonHandler(project_path)
        # Load model answers
        self.model_answers = self.handler.read_model_answers()
        
       
        
        if self.image_paths:
            self.current_index = 0
            self.show_current_image()
        else:
            self.lbl_image_info.setText("No processed images found")
            self.image_label.setText("No marked images available in results folder")
        
        self.update_navigation_buttons()
        self.update_review_button_state()
        self.project_loaded.emit(project_path)

    def show_current_image(self):
        """Display the current processed image with results"""
        if not self.image_paths:
            return
            
        image_path = self.image_paths[self.current_index]
        filename = os.path.basename(image_path)
        
        try:
            # Load and display image
            pixmap = QPixmap(image_path)
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            ))
            
            # Update image info
            self.lbl_image_info.setText(
                f"Image {self.current_index + 1}/{len(self.image_paths)}\n"
                f"{filename}"
            )
            
            # Update review button state
            self.update_review_button_state()
                
        except Exception as e:
            self.image_label.setText(f"Error loading image: {str(e)}")
            self.lbl_results_info.setText("")

    def show_next_image(self):
        """Navigate to the next image"""
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.show_current_image()
        self.update_navigation_buttons()

    def show_previous_image(self):
        """Navigate to the previous image"""
        if self.current_index > 0:
            self.current_index -= 1
            self.show_current_image()
        self.update_navigation_buttons()

    def update_navigation_buttons(self):
        """Update button states based on current position"""
        has_images = len(self.image_paths) > 0
        self.btn_prev.setEnabled(has_images and self.current_index > 0)
        self.btn_next.setEnabled(has_images and self.current_index < len(self.image_paths) - 1)
        self.btn_change_answer.setEnabled(has_images)
        self.btn_download_results.setEnabled(has_images)

    def update_review_button_state(self):
        """Update the Mark as Reviewed button state"""
        if not self.image_paths:
            self.btn_mark_reviewed.setEnabled(False)
            return
            
        current_image = os.path.basename(self.image_paths[self.current_index])
        is_reviewed = current_image in self.reviewed_images
        
        self.btn_mark_reviewed.setEnabled(not is_reviewed)
        self.btn_mark_reviewed.setText(
            "✓ Reviewed" if is_reviewed else "✓ Mark as Reviewed"
        )

    def mark_as_reviewed(self):
        """Mark current image as reviewed"""
        if not self.image_paths:
            return
            
        current_image = os.path.basename(self.image_paths[self.current_index])
        self.reviewed_images.add(current_image)
        self.update_review_button_state()
        image_path = os.path.join(
                self.project_path, "results", current_image
            )
        if not os.path.exists(image_path):
                raise FileNotFoundError(f"Original image not found: {image_path}")
            
        img = cv2.imread(image_path)
        try:
            reviewed_img = utils.draw_stamp(img, input_name="First Examiner", position=(25, 100), color=(0, 0, 255))
            cv2.imwrite(image_path, reviewed_img)

            # Update the displayed image
            self.image_paths[self.current_index] = image_path
            self.show_current_image()

            # Update db
            self.handler.mark_for_review(current_image, True)
        except Exception as e:
            print("Error occurs when draw stamps", e)

    def change_detected_answer(self):
        """Change the detected answer for selected question"""
        if not self.image_paths:
            return
        # Disable button to prevent multiple clicks
        self.btn_change_answer.setEnabled(False)  # Disable button during processing

        current_image = os.path.basename(self.image_paths[self.current_index])
        question_num = int(self.question_combo.currentText())
        new_answer = int(self.answer_combo.currentText())
        
        # Update the answer in the db
        try:
            new_detected_answers = self.handler.update_correction(
                current_image,
                question_num-1,  # Convert to 0-based index
                new_answer+1  # Detected answers saved as 1->2 2->3 3->4 4->5 and no answers saved as -1
            )

            # Get original image from original_images folder
            original_image_path = os.path.join(
                self.project_path, "original_images", current_image
            )
            if not os.path.exists(original_image_path):
                raise FileNotFoundError(f"Original image not found: {original_image_path}")
            
            img = cv2.imread(original_image_path)
            # Update the image with new answer
            result_img = utils.process_omr_sheet_without_model(
                img,
                new_detected_answers,
                self.model_answers
            )
            # Save the updated image back to results folder
            result_image_path = os.path.join(
                self.project_path, "results", current_image
            )
            cv2.imwrite(result_image_path, result_img)

            # Update the displayed image
            self.image_paths[self.current_index] = result_image_path
            self.show_current_image()

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
            self.btn_change_answer.setEnabled(True)

        # Enable button after processing
        self.btn_change_answer.setEnabled(True)

    def download_results(self):
        """Handle downloading of results to selected directory"""
        if not self.project_path:
            QMessageBox.warning(self, "No Project", "Please load a project first")
            return
            
        # Let user select download directory
        download_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Download Directory",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if not download_dir:
            return
            
        try:
            #get the project title using project path
            p_title=self.project_path.split(os.sep)[-1]
            # Create results directory in download location
            dest_dir = os.path.join(download_dir, p_title + "marked_sheets")
            os.makedirs(dest_dir, exist_ok=True)
            
            # Copy all result images
            results_src = os.path.join(self.project_path, "results")
            if os.path.exists(results_src):
                for file in os.listdir(results_src):
                    src_file = os.path.join(results_src, file)
                    if os.path.isfile(src_file):
                        shutil.copy2(src_file, dest_dir)
            
            # Copy JSON data files
            json_files = [f for f in os.listdir(self.project_path) if f.endswith('.json')]
            for json_file in json_files:
                src_file = os.path.join(self.project_path, json_file)
                shutil.copy2(src_file, dest_dir)

            #create excel sheet
            self.handler.export_to_excel(os.path.join(dest_dir, p_title+"result.xlsx"))

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
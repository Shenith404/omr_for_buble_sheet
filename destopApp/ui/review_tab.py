import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QSizePolicy, QFileDialog,
    QComboBox, QSpacerItem, QMessageBox
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt, Signal
import db  # Assuming db is a module for handling JSON data

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
        self.handler=None
        
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
        
        # Add a button to switch to Processing tab if needed
        self.btn_edit_processing = QPushButton("Edit Processing")
        self.btn_edit_processing.setStyleSheet("font-weight: bold;")
        info_layout.addWidget(self.btn_edit_processing)
        
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
        
        # Question Selection Group
        question_group = QGroupBox("Answer Correction")
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
        self.btn_edit_processing.clicked.connect(
            lambda: self.navigation_requested.emit(1)
        )
        self.btn_mark_reviewed.clicked.connect(self.mark_as_reviewed)
        self.btn_change_answer.clicked.connect(self.change_detected_answer)

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


        #load db
        self.handler= db.OMRJsonHandler(project_path)
        
        # Load reviewed status
        review_file = os.path.join(project_path, "review_status.json")
        if os.path.exists(review_file):
            try:
                with open(review_file, 'r') as f:
                    self.reviewed_images = set(json.load(f))
            except:
                pass
        
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
        self.btn_edit_processing.setEnabled(has_images)

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
        
        # Save reviewed status
        if self.project_path:
            review_file = os.path.join(self.project_path, "review_status.json")
            with open(review_file, 'w') as f:
                json.dump(list(self.reviewed_images), f)

    def change_detected_answer(self):
        """Change the detected answer for selected question"""
        if not self.image_paths:
            return
        #disable button to prevent multiple clicks
        self.btn_change_answer.setEnabled(False)  # Disable button during processing

        current_image = os.path.basename(self.image_paths[self.current_index])
        question_num = int(self.question_combo.currentText())
        new_answer = int(self.answer_combo.currentText())
        
        # Emit signal to notify about answer change
        #self.answer_modified.emit(current_image, question_num, new_answer)

        #update the answer in the db
        try:
            self.handler.update_correction(
                current_image,
                question_num-1,  # Convert to 0-based index
                new_answer+1  #detected answers saved as  1->2 2->3 3->4 4->5 and no answers saved as -1

            )

            #get original image from original_images folder
            original_image_path = os.path.join(
                self.project_path, "original_images", current_image
            )
            if not os.path.exists(original_image_path):
                raise FileNotFoundError(f"Original image not found: {original_image_path}")
            # Update the image with new answer
            

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





        #enable button after processing
        self.btn_change_answer.setEnabled(True)

    def resizeEvent(self, event):
        """Handle window resize to maintain image display"""
        super().resizeEvent(event)
        if self.image_paths:
            self.show_current_image()
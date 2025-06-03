import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QSizePolicy, QFileDialog
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt, Signal
import db  # Assuming db is a module for handling JSON data

class ReviewTab(QWidget):
    """Enhanced Review Tab with project loading and 70-30 split layout"""
    
    # Signals
    project_loaded = Signal(str)
    navigation_requested = Signal(int)  # Request to switch to Processing tab with index
    
    def __init__(self):
        super().__init__()
        self.project_path = None
        self.image_paths = []
        self.current_index = 0
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Initialize UI with 70-30 horizontal split layout"""
        main_layout = QHBoxLayout()
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
        
        # Right Panel (30%) - Empty for now (can add additional info later)
        right_panel = QWidget()
        right_panel.setStyleSheet("background-color: #f8f8f8;")
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(QLabel("Additional Information"))
        right_layout.addStretch()  # Push content to top
        
        # Add panels to main layout
        main_layout.addWidget(left_panel, 7)  # 70% width
        main_layout.addWidget(right_panel, 3)  # 30% width
        
        self.setLayout(main_layout)
        self.update_navigation_buttons()

    def setup_connections(self):
        """Connect all signals and slots"""
        self.btn_select_project.clicked.connect(self.select_project)
        self.btn_prev.clicked.connect(self.show_previous_image)
        self.btn_next.clicked.connect(self.show_next_image)
        self.btn_edit_processing.clicked.connect(
            lambda: self.navigation_requested.emit(1)  # Assuming Processing tab is index 1
        )

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
        
        # Load processed images from results folder
        results_dir = os.path.join(project_path, "results")
        if os.path.exists(results_dir):
            self.image_paths = [
                os.path.join(results_dir, f)
                for f in sorted(os.listdir(results_dir))
                if f.lower().startswith("marked_") and 
                f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ]
        
        # Load corresponding answers if available
        self.answers = {}
        json_handler = db.OMRJsonHandler(project_path)
        # if json_handler.data:
        #     for filename, data in json_handler.data.items():
        #         if filename.startswith("marked_"):
        #             self.answers[filename] = data
        
        if self.image_paths:
            self.current_index = 0
            self.show_current_image()
        else:
            self.lbl_image_info.setText("No processed images found")
            self.image_label.setText("No marked images available in results folder")
        
        self.update_navigation_buttons()
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
            
            # Show answer information if available
            if filename in self.answers:
                answers = self.answers[filename].get('answers', [])
                marks = self.answers[filename].get('total_marks', 0)
                self.lbl_results_info.setText(
                    f"Detected Answers: {answers}\n"
                    f"Total Marks: {marks}/50"
                )
            else:
                self.lbl_results_info.setText("No answer data available")
                
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

    def resizeEvent(self, event):
        """Handle window resize to maintain image display"""
        super().resizeEvent(event)
        if self.image_paths:
            self.show_current_image()
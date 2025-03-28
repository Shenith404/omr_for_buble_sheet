from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QComboBox, QFileDialog, 
    QLineEdit, QMessageBox, QGridLayout
)
from PySide6.QtGui import QPixmap, QImage, QIcon
from PySide6.QtCore import Qt, Signal, QTimer
import os
import cv2

class ProjectTab(QWidget):
    project_created = Signal(str)
    project_opened = Signal(str)
    images_added = Signal(list)
    
    def __init__(self):
        super().__init__()
        self.current_project = None
        self.webcam_active = False
        self.cap = None
        self.webcam_timer = QTimer()
        self.setup_ui()
        
    def setup_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Project Management
        self.setup_project_management_ui()
        
        # Image Source
        self.setup_image_source_ui()
        
        # Image Preview
        self.setup_image_preview_ui()
        
        main_layout.addWidget(self.project_group)
        main_layout.addWidget(self.source_group)
        main_layout.addWidget(self.preview_group)
        
        self.setLayout(main_layout)

    def setup_project_management_ui(self):
        self.project_group = QGroupBox("Project Management")
        layout = QGridLayout()
        
        # Project Name
        self.project_name_input = QLineEdit()
        self.project_name_input.setPlaceholderText("Enter project name")
        
        # Location
        self.location_label = QLabel("No location selected")
        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self.select_location)
        
        # Buttons
        self.btn_create = QPushButton("Create Project")
        self.btn_create.setIcon(QIcon.fromTheme("document-new"))
        self.btn_create.setStyleSheet("background-color: #4CAF50; color: white;")
        self.btn_create.clicked.connect(self.create_project)
        
        self.btn_open = QPushButton("Open Project")
        self.btn_open.setIcon(QIcon.fromTheme("document-open"))
        self.btn_open.setStyleSheet("background-color: #2196F3; color: white;")
        self.btn_open.clicked.connect(self.open_project_dialog)
        
        layout.addWidget(QLabel("Project Name:"), 0, 0)
        layout.addWidget(self.project_name_input, 0, 1, 1, 2)
        layout.addWidget(QLabel("Project Location:"), 1, 0)
        layout.addWidget(self.location_label, 1, 1)
        layout.addWidget(self.btn_browse, 1, 2)
        layout.addWidget(self.btn_create, 2, 0, 1, 3)
        layout.addWidget(self.btn_open, 3, 0, 1, 3)
        
        self.project_group.setLayout(layout)

    def setup_image_source_ui(self):
        self.source_group = QGroupBox("Image Source")
        layout = QHBoxLayout()
        
        self.source_combo = QComboBox()
        self.source_combo.addItems(["File", "Webcam"])
        self.source_combo.currentIndexChanged.connect(self.toggle_source)
        
        self.btn_add = QPushButton("Add Images (Copy)")
        self.btn_add.setIcon(QIcon.fromTheme("list-add"))
        self.btn_add.setStyleSheet("background-color: #2196F3; color: white;")
        self.btn_add.clicked.connect(self.add_images)
        self.btn_add.setEnabled(False)
        
        self.btn_link = QPushButton("Link Images (Reference)")
        self.btn_link.setIcon(QIcon.fromTheme("emblem-symbolic-link"))
        self.btn_link.setStyleSheet("background-color: #9C27B0; color: white;")
        self.btn_link.clicked.connect(self.link_images)
        self.btn_link.setEnabled(False)
        
        self.btn_capture = QPushButton("Capture Image")
        self.btn_capture.setIcon(QIcon.fromTheme("camera-photo"))
        self.btn_capture.setStyleSheet("background-color: #FF9800; color: white;")
        self.btn_capture.clicked.connect(self.capture_webcam_image)
        self.btn_capture.hide()
        
        layout.addWidget(self.source_combo)
        layout.addWidget(self.btn_add)
        layout.addWidget(self.btn_link)
        layout.addWidget(self.btn_capture)
        
        self.source_group.setLayout(layout)
        self.source_group.setEnabled(False)

    def setup_image_preview_ui(self):
        self.preview_group = QGroupBox("Image Preview")
        layout = QVBoxLayout()
        
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("""
            background-color: #f0f0f0;
            border: 2px solid #ccc;
            min-height: 300px;
        """)
        
        self.status_label = QLabel("No project loaded")
        self.status_label.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(self.preview_label)
        layout.addWidget(self.status_label)
        
        self.preview_group.setLayout(layout)

    def select_location(self):
        location = QFileDialog.getExistingDirectory(self, "Select Project Location")
        if location:
            self.location_label.setText(location)

    def create_project(self):
        project_name = self.project_name_input.text().strip()
        location = self.location_label.text()
        
        if not project_name:
            QMessageBox.warning(self, "Error", "Please enter a project name")
            return
            
        if location == "No location selected":
            QMessageBox.warning(self, "Error", "Please select a location")
            return
            
        project_path = os.path.join(location, project_name)
        
        try:
            os.makedirs(project_path, exist_ok=True)
            os.makedirs(os.path.join(project_path, "original_images"), exist_ok=True)
            open(os.path.join(project_path, "image_references.txt"), 'w').close()
            
            self.current_project = project_path
            self.source_group.setEnabled(True)
            self.btn_link.setEnabled(True)
            self.status_label.setText(f"Project created: {project_path}")
            self.project_created.emit(project_path)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not create project:\n{str(e)}")

    def open_project_dialog(self):
        project_path = QFileDialog.getExistingDirectory(self, "Open Project")
        if project_path:
            self.load_project(project_path)

    def load_project(self, project_path):
        if not os.path.exists(os.path.join(project_path, "image_references.txt")):
            QMessageBox.warning(self, "Error", "Not a valid OMR Scanner project")
            return
            
        self.current_project = project_path
        self.project_name_input.setText(os.path.basename(project_path))
        self.location_label.setText(os.path.dirname(project_path))
        
        self.source_group.setEnabled(True)
        self.btn_link.setEnabled(True)
        self.status_label.setText(f"Project loaded: {project_path}")
        self.project_opened.emit(project_path)

    def toggle_source(self, index):
        if index == 0:  # File
            self.btn_capture.hide()
            self.btn_add.show()
            self.btn_link.show()
            self.stop_webcam()
        else:  # Webcam
            self.btn_add.hide()
            self.btn_link.hide()
            self.btn_capture.show()
            self.start_webcam()

    def start_webcam(self):
        if not self.webcam_active:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.webcam_active = True
                self.webcam_timer.timeout.connect(self.update_webcam_preview)
                self.webcam_timer.start(30)
                self.status_label.setText("Webcam active - ready to capture")
            else:
                self.source_combo.setCurrentIndex(0)
                QMessageBox.warning(self, "Error", "Could not open webcam")

    def stop_webcam(self):
        if self.webcam_active:
            self.webcam_timer.stop()
            self.webcam_active = False
            if self.cap:
                self.cap.release()
                self.cap = None
            self.preview_label.clear()
            self.preview_label.setText("Webcam inactive")

    def update_webcam_preview(self):
        if self.webcam_active and self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image).scaled(
                    self.preview_label.width(),
                    self.preview_label.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.preview_label.setPixmap(pixmap)

    def capture_webcam_image(self):
        if hasattr(self, 'current_frame') and self.current_frame is not None:
            saved_path = self.save_image(self.current_frame)
            if saved_path:
                self.images_added.emit([saved_path])
                self.status_label.setText("Webcam image captured and saved")

    def add_images(self):
        if not self.current_project:
            QMessageBox.warning(self, "Error", "No project loaded")
            return
            
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", "",
            "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if files:
            saved_paths = []
            for file in files:
                try:
                    img = cv2.imread(file)
                    if img is not None:
                        saved_path = self.save_image(img, os.path.basename(file))
                        saved_paths.append(saved_path)
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Could not load {file}:\n{str(e)}")
            
            if saved_paths:
                self.images_added.emit(saved_paths)
                self.status_label.setText(f"Added {len(saved_paths)} images to project")
                self.show_image(saved_paths[0])

    def link_images(self):
        if not self.current_project:
            QMessageBox.warning(self, "Error", "No project loaded")
            return
            
        folder = QFileDialog.getExistingDirectory(self, "Select Folder with Images")
        if not folder:
            return
            
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        image_paths = []
        
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(image_extensions):
                    image_paths.append(os.path.join(root, file))
        
        if not image_paths:
            QMessageBox.warning(self, "No Images", "No supported images found")
            return
        
        ref_file = os.path.join(self.current_project, "image_references.txt")
        with open(ref_file, 'w') as f:
            f.write('\n'.join(image_paths))
        
        self.images_added.emit(image_paths)
        self.status_label.setText(f"Linked {len(image_paths)} images")
        
        if image_paths:
            self.show_image(image_paths[0])

    def save_image(self, image, filename=None):
        if not self.current_project:
            return None
            
        images_dir = os.path.join(self.current_project, "original_images")
        os.makedirs(images_dir, exist_ok=True)
        
        if filename is None:
            filename = f"capture_{len(os.listdir(images_dir)) + 1}.jpg"
        
        base, ext = os.path.splitext(filename)
        counter = 1
        while os.path.exists(os.path.join(images_dir, filename)):
            filename = f"{base}_{counter}{ext}"
            counter += 1
        
        save_path = os.path.join(images_dir, filename)
        cv2.imwrite(save_path, image)
        return save_path

    def show_image(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image).scaled(
                    self.preview_label.width(),
                    self.preview_label.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.preview_label.setPixmap(pixmap)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not display image:\n{str(e)}")
    
    def set_project_path(self, project_path):
        """Public interface to set project path"""
        self.load_project(project_path)

    def closeEvent(self, event):
        self.stop_webcam()
        event.accept()
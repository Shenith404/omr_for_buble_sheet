import os
import cv2
import traceback
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QComboBox, QFileDialog,
    QLineEdit, QMessageBox, QGridLayout, QSizePolicy
)
from PySide6.QtGui import QPixmap, QImage, QIcon, QFont, QColor
from PySide6.QtCore import Qt, Signal, QTimer, QSize
import utils  # Assuming utils is a module with required functions
import platform

class ProjectTab(QWidget):
    project_created = Signal(str)
    project_opened = Signal(str)
    images_added = Signal(list)

    def __init__(self):
        super().__init__()
        self.current_project = None
        self.webcam_active = False
        self.selected_webcam_index = 0
        self.available_cameras = []
        self.cap = None
        self.webcam_timer = QTimer()
        self.captured_frame = None
        self.preview_width = 640
        self.preview_height = 480

        self.setup_ui()

    def setup_ui(self):
        """Initialize all UI components"""
        try:
            # Apply dark theme to the entire widget
            self.setStyleSheet("""
                QWidget {
                    background-color: #181818;
                    color: #e8e8e8;
                    font-family: 'Segoe UI', 'SF Pro Display', 'Inter', 'Roboto', sans-serif;
                }
                QMessageBox {
                    background-color: #2a2a2a;
                    color: #e8e8e8;
                }
                QMessageBox QPushButton {
                    background: linear-gradient(180deg, #0078d4 0%, #005a9e 100%);
                    border: 1px solid #004578;
                    border-radius: 6px;
                    color: #ffffff;
                    padding: 8px 16px;
                    min-width: 80px;
                }
            """)

            # Main layout
            main_layout = QVBoxLayout()
            main_layout.setContentsMargins(20, 20, 20, 20)
            main_layout.setSpacing(20)

            # Setup components
            self.setup_project_management_ui()
            self.setup_image_source_ui()
            self.setup_image_preview_ui()

            # Add components to main layout
            main_layout.addWidget(self.project_group)
            main_layout.addWidget(self.source_group)
            main_layout.addWidget(self.preview_group, stretch=1)

            self.setLayout(main_layout)
            self.setMinimumSize(900, 750)

        except Exception as e:
            self.show_error("UI Setup Failed", f"Failed to initialize UI: {str(e)}")

    def setup_project_management_ui(self):
        """Setup project creation/opening controls"""
        try:
            self.project_group = QGroupBox("Project Management")
            self.project_group.setStyleSheet("""
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
            """)
            layout = QGridLayout()
            layout.setSpacing(15)

            # Project Name
            self.project_name_input = QLineEdit()
            self.project_name_input.setPlaceholderText("Enter project name")
            self.project_name_input.setMinimumWidth(250)
            self.project_name_input.setStyleSheet("""
                QLineEdit {
                    background: #2a2a2a;
                    border: 1px solid #404040;
                    border-radius: 6px;
                    color: #e8e8e8;
                    font-size: 13px;
                    padding: 10px 12px;
                    selection-background-color: #0078d4;
                }
                QLineEdit:focus {
                    border-color: #0078d4;
                    background: #323232;
                }
                QLineEdit::placeholder {
                    color: #888888;
                }
            """)

            # Location
            self.location_label = QLabel("No location selected")
            self.location_label.setWordWrap(True)
            self.location_label.setStyleSheet("""
                QLabel {
                    background: #2a2a2a;
                    border: 1px solid #404040;
                    border-radius: 6px;
                    color: #cccccc;
                    padding: 10px 12px;
                    font-style: italic;
                }
            """)
            
            self.btn_browse = QPushButton("Browse...")
            self.btn_browse.clicked.connect(self.select_location)
            self.btn_browse.setStyleSheet("""
                QPushButton {
                    background: linear-gradient(180deg, #505050 0%, #404040 100%);
                    border: 1px solid #606060;
                    border-radius: 6px;
                    color: #ffffff;
                    font-size: 13px;
                    font-weight: 500;
                    padding: 10px 16px;
                    min-width: 80px;
                }
                QPushButton:hover {
                    background: linear-gradient(180deg, #606060 0%, #505050 100%);
                    border-color: #707070;
                }
                QPushButton:pressed {
                    background: linear-gradient(180deg, #404040 0%, #303030 100%);
                }
            """)

            # Buttons
            self.btn_create = self.create_styled_button(
                "Create Project", "#28a745", "document-new")
            self.btn_create.clicked.connect(self.create_project)

            self.btn_open = self.create_styled_button(
                "Open Project", "#007bff", "document-open")
            self.btn_open.clicked.connect(self.open_project_dialog)

            # Labels with proper styling
            name_label = QLabel("Project Name:")
            name_label.setStyleSheet("QLabel { color: #e8e8e8; font-weight: 500; }")
            
            location_label_header = QLabel("Project Location:")
            location_label_header.setStyleSheet("QLabel { color: #e8e8e8; font-weight: 500; }")

            # Add to layout
            layout.addWidget(name_label, 0, 0)
            layout.addWidget(self.project_name_input, 0, 1, 1, 2)
            layout.addWidget(location_label_header, 1, 0)
            layout.addWidget(self.location_label, 1, 1)
            layout.addWidget(self.btn_browse, 1, 2)
            layout.addWidget(self.btn_create, 2,2)
            layout.addWidget(self.btn_open, 2, 1)

            self.project_group.setLayout(layout)

        except Exception as e:
            self.show_error("Setup Error", f"Failed to setup project UI: {str(e)}")

    def setup_image_source_ui(self):
        """Setup image source selection controls"""
        try:
            self.source_group = QGroupBox("Image Source")
            self.source_group.setStyleSheet("""
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
            """)
            layout = QHBoxLayout()
            layout.setSpacing(15)

            # Source selection
            self.source_combo = QComboBox()
            self.source_combo.addItems(["File", "Webcam"])
            self.source_combo.currentIndexChanged.connect(self.toggle_source)
            self.source_combo.setMinimumWidth(150)
            self.source_combo.setStyleSheet("""
                QComboBox {
                    background: #2a2a2a;
                    border: 1px solid #404040;
                    border-radius: 6px;
                    color: #e8e8e8;
                    font-size: 13px;
                    padding: 8px 12px;
                    min-width: 120px;
                    font-weight: 500;
                }
                QComboBox:focus {
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
            """)

            # webcam selection
            self.webcam_selection_box = QComboBox()
            self.webcam_selection_box.addItems(self.available_cameras)
            self.webcam_selection_box.currentIndexChanged.connect(self.toggle_source)
            self.webcam_selection_box.setMinimumWidth(150)
            self.webcam_selection_box.setStyleSheet("""
                QComboBox {
                    background: #2a2a2a;
                    border: 1px solid #404040;
                    border-radius: 6px;
                    color: #e8e8e8;
                    font-size: 13px;
                    padding: 8px 12px;
                    min-width: 120px;
                    font-weight: 500;
                }
                QComboBox:focus {
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
            """)
            self.webcam_selection_box.hide()
            
            # Action buttons
            self.btn_add = self.create_styled_button(
                "Add Images", "#007bff", "list-add")
            self.btn_add.clicked.connect(self.add_images)
            self.btn_add.setEnabled(False)

            self.btn_link = self.create_styled_button(
                "Link Images", "#6f42c1", "emblem-symbolic-link")
            self.btn_link.clicked.connect(self.link_images)
            self.btn_link.setEnabled(False)

            self.btn_capture = self.create_styled_button(
                "Capture", "#fd7e14", "camera-photo")
            self.btn_capture.clicked.connect(self.capture_webcam_image)
            self.btn_capture.hide()

            self.btn_save = self.create_styled_button(
                "Save Image", "#28a745", "document-save")
            self.btn_save.clicked.connect(self.save_captured_image)
            self.btn_save.hide()

            # Add to layout
            layout.addWidget(self.source_combo)
            layout.addWidget(self.webcam_selection_box)
            layout.addWidget(self.btn_add)
            layout.addWidget(self.btn_link)
            layout.addWidget(self.btn_capture)
            layout.addWidget(self.btn_save)
            layout.addStretch()  # Add stretch to push buttons to the left

            self.source_group.setLayout(layout)
            self.source_group.setEnabled(False)

        except Exception as e:
            self.show_error("Setup Error", f"Failed to setup source UI: {str(e)}")

    def setup_image_preview_ui(self):
        """Setup image preview area"""
        try:
            self.preview_group = QGroupBox("Image Preview")
            self.preview_group.setStyleSheet("""
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
            """)
            layout = QVBoxLayout()
            layout.setSpacing(15)

            # Preview label
            self.preview_label = QLabel()
            self.preview_label.setAlignment(Qt.AlignCenter)
            self.preview_label.setMinimumSize(self.preview_width, self.preview_height)
            self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.preview_label.setStyleSheet("""
                QLabel {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #2a2a2a, stop:1 #1e1e1e);
                    border: 2px dashed #505050;
                    border-radius: 8px;
                    color: #888888;
                    font-size: 14px;
                    font-weight: 500;
                    padding: 20px;
                }
                QLabel:hover {
                    border-color: #0078d4;
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #323232, stop:1 #262626);
                }
            """)
            self.preview_label.setText("📷 No image to display\n\nDrag & drop images here or use the controls above")

            # Status label
            self.status_label = QLabel("No project loaded")
            self.status_label.setAlignment(Qt.AlignCenter)
            self.status_label.setStyleSheet("""
                QLabel {
                    background: #262626;
                    border: 1px solid #404040;
                    border-radius: 6px;
                    color: #cccccc;
                    font-style: italic;
                    font-size: 12px;
                    padding: 8px 12px;
                    font-weight: 400;
                }
            """)

            # Add to layout
            layout.addWidget(self.preview_label, stretch=1)
            layout.addWidget(self.status_label)

            self.preview_group.setLayout(layout)

        except Exception as e:
            self.show_error("Setup Error", f"Failed to setup preview UI: {str(e)}")

    def create_styled_button(self, text, color, icon_name):
        """Helper to create consistently styled buttons with modern dark theme"""
        btn = QPushButton(text)
        # Note: Icons might not display in dark theme, focusing on color-coded buttons
        
        # Convert hex colors to modern gradient styles
        color_map = {
            "#28a745": ("linear-gradient(180deg, #28a745 0%, #1e7e34 100%)", "#1e7e34"),  # Success Green
            "#007bff": ("linear-gradient(180deg, #007bff 0%, #0056b3 100%)", "#0056b3"),  # Primary Blue  
            "#6f42c1": ("linear-gradient(180deg, #6f42c1 0%, #5a2d8c 100%)", "#5a2d8c"),  # Purple
            "#fd7e14": ("linear-gradient(180deg, #fd7e14 0%, #e55a00 100%)", "#e55a00"),  # Orange
            "#dc3545": ("linear-gradient(180deg, #dc3545 0%, #c82333 100%)", "#c82333"),  # Danger Red
        }
        
        gradient, pressed_color = color_map.get(color, (f"linear-gradient(180deg, {color} 0%, {self.darken_color(color)} 100%)", self.darken_color(color)))
        
        btn.setStyleSheet(f"""
            QPushButton {{
                background: {gradient};
                border: 1px solid {self.darken_color(color)};
                border-radius: 6px;
                color: #ffffff;
                font-size: 13px;
                font-weight: 500;
                padding: 10px 20px;
                min-width: 120px;
                min-height: 16px;
            }}
            QPushButton:hover {{
                background: {self.lighten_color(gradient)};
                border-color: {color};
            }}
            QPushButton:pressed {{
                background: linear-gradient(180deg, {pressed_color} 0%, {self.darken_color(pressed_color)} 100%);
                border-color: {self.darken_color(pressed_color)};
            }}
            QPushButton:disabled {{
                background: linear-gradient(180deg, #3a3a3a 0%, #2a2a2a 100%);
                border-color: #525252;
                color: #7a7a7a;
            }}
        """)
        return btn

    def darken_color(self, hex_color, factor=0.8):
        """Darken a hex color for hover effects"""
        try:
            color = QColor(hex_color)
            return color.darker(100 + int(100 * (1 - factor))).name()
        except:
            return hex_color

    def lighten_color(self, gradient_or_color, factor=1.2):
        """Lighten a color or modify a gradient for hover effects"""
        if "linear-gradient" in str(gradient_or_color):
            # For gradients, extract colors and lighten them
            return str(gradient_or_color).replace("28a745", "34d058").replace("007bff", "1a8cff").replace("6f42c1", "8b5cf6").replace("fd7e14", "ff9500").replace("dc3545", "ff4757")
        else:
            try:
                color = QColor(gradient_or_color)
                return color.lighter(int(100 * factor)).name()
            except:
                return gradient_or_color

    def validate_image(self, image):
        """Validate the captured image before saving"""
        # Step 1: Preprocessing with optimized operations


        try:
            img = cv2.resize(image, (1025, 760))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 1)
            edges = cv2.Canny(blur, 10, 50)
            

            # Step 2: Contour Detection with area filtering
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            rects = utils.rectContour(contours)
        except:
            return False
        
        if not rects:
            return False

        # Step 3: Perspective Transform with error checking
        biggest = utils.getCornerPoints(rects[0])
        if biggest.size == 0:
            return False
        return True

    def select_location(self):
        """Handle location selection via file dialog"""
        try:
            location = QFileDialog.getExistingDirectory(
                self, 
                "Select Project Location",
                os.path.expanduser("~"),
                QFileDialog.ShowDirsOnly
            )
            if location:
                self.location_label.setText(location)
                self.location_label.setToolTip(location)
        except Exception as e:
            self.show_error("Location Error", f"Failed to select location: {str(e)}")

    def create_project(self):
        """Create a new project directory structure"""
        try:
            project_name = self.project_name_input.text().strip()
            if not project_name:
                QMessageBox.warning(self, "Input Error", "Please enter a project name")
                return

            location = self.location_label.text()
            if location == "No location selected":
                QMessageBox.warning(self, "Input Error", "Please select a location")
                return

            project_path = os.path.join(location, project_name)
            
            # Create directory structure
            os.makedirs(project_path, exist_ok=True)
            os.makedirs(os.path.join(project_path, "original_images"), exist_ok=True)
            
            # Create empty references file
            with open(os.path.join(project_path, "image_references.txt"), 'w') as f:
                f.write("")

            self.current_project = project_path
            self.source_group.setEnabled(True)
            self.btn_link.setEnabled(True)
            self.btn_add.setEnabled(True)
            self.status_label.setText(f"Project created: {project_path}")
            self.project_created.emit(project_path)

        except FileExistsError:
            QMessageBox.warning(self, "Project Exists", "A project with this name already exists")
        except PermissionError:
            self.show_error("Permission Error", "You don't have permission to create a project here")
        except Exception as e:
            self.show_error("Project Error", f"Failed to create project: {str(e)}")

    def open_project_dialog(self):
        """Open an existing project"""
        try:
            project_path = QFileDialog.getExistingDirectory(
                self,
                "Open Project",
                os.path.expanduser("~"),
                QFileDialog.ShowDirsOnly
            )
            if project_path:
                self.load_project(project_path)
        except Exception as e:
            self.show_error("Open Error", f"Failed to open project: {str(e)}")

    def load_project(self, project_path):
        """Load an existing project"""
        try:
            if not os.path.exists(project_path):
                raise FileNotFoundError("Project directory not found")

            if not os.path.exists(os.path.join(project_path, "image_references.txt")):
                raise ValueError("Not a valid project directory")

            self.current_project = project_path
            self.project_name_input.setText(os.path.basename(project_path))
            self.location_label.setText(os.path.dirname(project_path))
            
            self.source_group.setEnabled(True)
            self.btn_link.setEnabled(True)
            self.btn_add.setEnabled(True)
            self.status_label.setText(f"Project loaded: {project_path}")
            self.project_opened.emit(project_path)

        except FileNotFoundError:
            QMessageBox.warning(self, "Not Found", "The specified project directory doesn't exist")
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Project", str(e))
        except Exception as e:
            self.show_error("Load Error", f"Failed to load project: {str(e)}")

    def toggle_source(self, index):
        """Toggle between file and webcam source"""
        try:
            if index == 0:  # File
                self.btn_capture.hide()
                self.webcam_selection_box.hide()
                self.btn_save.hide()
                self.btn_add.show()
                self.btn_link.show()
                self.stop_webcam()
                self.preview_label.clear()
                self.preview_label.setText("Select images to preview")
                self.status_label.setText("Ready to add or link images")
            else:  # Webcam
                self.btn_add.hide()
                self.btn_link.hide()
                self.btn_capture.show()
                self.find_available_cameras()
                self.webcam_selection_box.show()
                self.start_webcam()
        except Exception as e:
            self.show_error("Source Error", f"Failed to switch source: {str(e)}")
            self.source_combo.setCurrentIndex(0)

    def start_webcam(self):
        """Initialize and start webcam capture"""
        try:
            if self.webcam_active:
                return

            self.cap = cv2.VideoCapture(self.selected_webcam_index)
            if not self.cap.isOpened():
                raise RuntimeError("Could not open webcam")

            self.webcam_active = True
            self.webcam_timer.timeout.connect(self.update_webcam_preview)
            self.webcam_timer.start(30)  # ~30 FPS
            self.status_label.setText("Webcam active - ready to capture")
            self.preview_label.setText("Initializing webcam...")

        except Exception as e:
            self.show_error("Webcam Error", f"Failed to start webcam: {str(e)}")
            self.stop_webcam()
            self.source_combo.setCurrentIndex(0)

    # In your find_available_cameras method:

    def find_available_cameras(self):
        """
        Finds and returns a list of available camera indices using a stable backend.
        Automatically selects the right backend based on OS and availability.
        """
        available_cameras = []

        # Automatically choose backend based on platform
        if platform.system() == "Windows":
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_VFW]  # Try these in order
        else:
            backends = [cv2.CAP_V4L2, cv2.CAP_ANY]  # Linux/macOS
        
        for backend in backends:
            for i in range(10):
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    available_cameras.append(i)
                    cap.release()
            # If we found at least one, stop trying further backends
            if available_cameras:
                break

        self.available_cameras = available_cameras
        print(f"✅ Available cameras: {available_cameras} (using backend {backend})")

    def stop_webcam(self):
        """Stop webcam capture and release resources"""
        try:
            if not self.webcam_active:
                return

            self.webcam_timer.stop()
            self.webcam_active = False
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.preview_label.clear()
            self.preview_label.setText("Webcam inactive")
        except Exception as e:
            self.show_error("Webcam Error", f"Failed to stop webcam: {str(e)}")

    def update_webcam_preview(self):
        """Update the preview with the current webcam frame"""
        try:
            if not self.webcam_active or self.cap is None:
                return

            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("Failed to capture frame")

            self.current_frame = frame
            self.display_image(frame)

        except Exception as e:
            self.show_error("Preview Error", f"Failed to update preview: {str(e)}")
            self.stop_webcam()
            self.source_combo.setCurrentIndex(0)

    def capture_webcam_image(self):
        """Capture the current webcam frame"""
        try:
            if not hasattr(self, 'current_frame') or self.current_frame is None:
                QMessageBox.warning(self, "Capture Error", "No frame available to capture")
                return

            self.captured_frame = self.current_frame.copy()
            
           
          
            # Directly save without preview
            self.save_captured_image()

        except Exception as e:
            self.show_error("Capture Error", f"Failed to capture image: {str(e)}")
            self.start_webcam()

    def display_image(self, image):
        """Display an image in the preview area"""
        try:
            if image is None or image.size == 0:
                raise ValueError("Invalid image data")

            # Convert color space
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w

            # Create QImage
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            if qt_image.isNull():
                raise ValueError("Failed to create QImage")

            # Convert to QPixmap and scale
            pixmap = QPixmap.fromImage(qt_image)
            if pixmap.isNull():
                raise ValueError("Failed to create QPixmap")

            scaled_pixmap = pixmap.scaled(
                self.preview_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            self.preview_label.setPixmap(scaled_pixmap)

        except Exception as e:
            self.show_error("Display Error", f"Failed to display image: {str(e)}")
            self.preview_label.setText("Failed to display image")

    def save_captured_image(self):
        """Save the captured image to the project"""
        if not self.validate_image(self.captured_frame):
            QMessageBox.warning(self, "Validation Error", "Image validation failed")
            return

        try:
            if self.captured_frame is None:
                QMessageBox.warning(self, "Save Error", "No image to save")
                return

            if not self.current_project:
                QMessageBox.warning(self, "Save Error", "No project selected")
                return

            saved_path = self.save_image(self.captured_frame)
            if not saved_path:
                raise RuntimeError("Failed to save image")

            self.images_added.emit([saved_path])
            
            # Show save confirmation
            QMessageBox.information(
                self,
                "Success",
                f"Image saved successfully to:\n{saved_path}",
                QMessageBox.Ok
            )
            
            # Reset UI
            self.reset_after_save()

        except Exception as e:
            self.show_error("Save Error", f"Failed to save image: {str(e)}")

    def add_images(self):
        """Add images by copying them to the project"""
        print("Adding images...")
        try:
            if not self.current_project:
                QMessageBox.warning(self, "Error", "No project loaded")
                return

            files, _ = QFileDialog.getOpenFileNames(
                self,
                "Select Images",
                os.path.expanduser("~"),
                "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
            )

            if not files:
                return

            saved_paths = []
            for file in files:
                try:
                    img = cv2.imread(file)
                    if img is None:
                        QMessageBox.warning(self, "Read Error", f"Could not read image: {file}")
                        continue

                    saved_path = self.save_image(img, os.path.basename(file))
                    if saved_path:
                        saved_paths.append(saved_path)
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to process {file}:\n{str(e)}")

            if saved_paths:
                self.images_added.emit(saved_paths)
                self.status_label.setText(f"Added {len(saved_paths)} images to project")
                self.show_image(saved_paths[0])

        except Exception as e:
            self.show_error("Add Error", f"Failed to add images: {str(e)}")

    def link_images(self):
        """Link images by referencing their original locations"""
        try:
            if not self.current_project:
                QMessageBox.warning(self, "Error", "No project loaded")
                return

            folder = QFileDialog.getExistingDirectory(
                self,
                "Select Folder with Images",
                os.path.expanduser("~")
            )

            if not folder:
                return

            image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
            image_paths = []
            
            for root, _, files in os.walk(folder):
                for file in files:
                    if file.lower().endswith(image_extensions):
                        image_paths.append(os.path.join(root, file))

            if not image_paths:
                QMessageBox.warning(self, "No Images", "No supported images found in selected folder")
                return

            ref_file = os.path.join(self.current_project, "image_references.txt")
            with open(ref_file, 'w') as f:
                f.write('\n'.join(image_paths))

            self.images_added.emit(image_paths)
            self.status_label.setText(f"Linked {len(image_paths)} images")
            
            if image_paths:
                self.show_image(image_paths[0])

        except Exception as e:
            self.show_error("Link Error", f"Failed to link images: {str(e)}")

    def save_image(self, image, filename=None):
        """Save image to project folder"""
        try:
            if not self.current_project:
                QMessageBox.warning(self, "Save Error", "No project selected")
                return None

            if image is None or image.size == 0:
                QMessageBox.warning(self, "Save Error", "No image data to save")
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
            if not cv2.imwrite(save_path, image):
                raise RuntimeError("OpenCV failed to write image")

            return save_path

        except Exception as e:
            self.show_error("Save Error", f"Failed to save image: {str(e)}")
            return None

    def show_image(self, image_path):
        """Display an image from file path"""
        try:
            if not os.path.exists(image_path):
                QMessageBox.warning(self, "Error", f"Image not found: {image_path}")
                return

            img = cv2.imread(image_path)
            if img is None:
                QMessageBox.warning(self, "Error", f"Could not read image: {image_path}")
                return

            self.display_image(img)

        except Exception as e:
            self.show_error("Display Error", f"Failed to show image: {str(e)}")

    def reset_after_save(self):
        """Reset UI after saving an image"""
        try:
            self.btn_save.hide()
            self.btn_capture.show()
            
            self.preview_label.clear()
            self.preview_label.setText("Ready for new capture")
            self.status_label.setText("Webcam ready")
            
            if self.source_combo.currentIndex() == 1:  # Webcam selected
                self.start_webcam()
                
        except Exception as e:
            self.show_error("Reset Error", f"Failed to reset UI: {str(e)}")

    def show_error(self, title, message):
        """Display detailed error message"""
        error_box = QMessageBox(self)
        error_box.setIcon(QMessageBox.Critical)
        error_box.setWindowTitle(title)
        error_box.setText(message)
        error_box.setDetailedText(traceback.format_exc())
        error_box.exec_()

    def set_project_path(self, project_path):
        """Public method to set project path programmatically"""
        try:
            if project_path and os.path.exists(project_path):
                self.load_project(project_path)
        except Exception as e:
            self.show_error("Load Error", f"Failed to set project path: {str(e)}")

    def closeEvent(self, event):
        """Clean up when closing the tab"""
        try:
            self.stop_webcam()
            if self.cap is not None:
                self.cap.release()
            event.accept()
        except Exception as e:
            self.show_error("Cleanup Error", f"Failed to clean up resources: {str(e)}")
            event.accept()
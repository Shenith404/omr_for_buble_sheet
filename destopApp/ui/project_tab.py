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
        self.captured_frame = None
        self.preview_width = 640
        self.preview_height = 480
        self.setup_ui()

    def setup_ui(self):
        """Initialize all UI components"""
        try:
            # Main layout
            main_layout = QVBoxLayout()
            main_layout.setContentsMargins(15, 15, 15, 15)
            main_layout.setSpacing(15)

            # Setup components
            self.setup_project_management_ui()
            self.setup_image_source_ui()
            self.setup_image_preview_ui()

            # Add components to main layout
            main_layout.addWidget(self.project_group)
            main_layout.addWidget(self.source_group)
            main_layout.addWidget(self.preview_group, stretch=1)

            self.setLayout(main_layout)
            self.setMinimumSize(800, 700)

        except Exception as e:
            self.show_error("UI Setup Failed", f"Failed to initialize UI: {str(e)}")

    def setup_project_management_ui(self):
        """Setup project creation/opening controls"""
        try:
            self.project_group = QGroupBox("Project Management")
            self.project_group.setFont(QFont("Arial", 10, QFont.Bold))
            layout = QGridLayout()
            layout.setSpacing(15)

            # Project Name
            self.project_name_input = QLineEdit()
            self.project_name_input.setPlaceholderText("Enter project name")
            self.project_name_input.setMinimumWidth(250)

            # Location
            self.location_label = QLabel("No location selected")
            self.location_label.setWordWrap(True)
            self.btn_browse = QPushButton("Browse...")
            self.btn_browse.clicked.connect(self.select_location)
            self.btn_browse.setStyleSheet("padding: 5px;")

            # Buttons
            self.btn_create = self.create_styled_button(
                "Create Project", "#4CAF50", "document-new")
            self.btn_create.clicked.connect(self.create_project)

            self.btn_open = self.create_styled_button(
                "Open Project", "#2196F3", "document-open")
            self.btn_open.clicked.connect(self.open_project_dialog)

            # Add to layout
            layout.addWidget(QLabel("Project Name:"), 0, 0)
            layout.addWidget(self.project_name_input, 0, 1, 1, 2)
            layout.addWidget(QLabel("Project Location:"), 1, 0)
            layout.addWidget(self.location_label, 1, 1)
            layout.addWidget(self.btn_browse, 1, 2)
            layout.addWidget(self.btn_create, 2, 0, 1, 3)
            layout.addWidget(self.btn_open, 3, 0, 1, 3)

            self.project_group.setLayout(layout)

        except Exception as e:
            self.show_error("Setup Error", f"Failed to setup project UI: {str(e)}")

    def setup_image_source_ui(self):
        """Setup image source selection controls"""
        try:
            self.source_group = QGroupBox("Image Source")
            self.source_group.setFont(QFont("Arial", 10, QFont.Bold))
            layout = QHBoxLayout()
            layout.setSpacing(10)

            # Source selection
            self.source_combo = QComboBox()
            self.source_combo.addItems(["File", "Webcam"])
            self.source_combo.currentIndexChanged.connect(self.toggle_source)
            self.source_combo.setMinimumWidth(150)

            # Action buttons
            self.btn_add = self.create_styled_button(
                "Add Images", "#2196F3", "list-add")
            self.btn_add.clicked.connect(self.add_images)
            self.btn_add.setEnabled(False)

            self.btn_link = self.create_styled_button(
                "Link Images", "#9C27B0", "emblem-symbolic-link")
            self.btn_link.clicked.connect(self.link_images)
            self.btn_link.setEnabled(False)

            self.btn_capture = self.create_styled_button(
                "Capture", "#FF9800", "camera-photo")
            self.btn_capture.clicked.connect(self.capture_webcam_image)
            self.btn_capture.hide()

            self.btn_save = self.create_styled_button(
                "Save Image", "#4CAF50", "document-save")
            self.btn_save.clicked.connect(self.save_captured_image)
            self.btn_save.hide()

            # Add to layout
            layout.addWidget(self.source_combo)
            layout.addWidget(self.btn_add)
            layout.addWidget(self.btn_link)
            layout.addWidget(self.btn_capture)
            layout.addWidget(self.btn_save)

            self.source_group.setLayout(layout)
            self.source_group.setEnabled(False)

        except Exception as e:
            self.show_error("Setup Error", f"Failed to setup source UI: {str(e)}")

    def setup_image_preview_ui(self):
        """Setup image preview area"""
        try:
            self.preview_group = QGroupBox("Image Preview")
            self.preview_group.setFont(QFont("Arial", 10, QFont.Bold))
            layout = QVBoxLayout()
            layout.setSpacing(10)

            # Preview label
            self.preview_label = QLabel()
            self.preview_label.setAlignment(Qt.AlignCenter)
            self.preview_label.setMinimumSize(self.preview_width, self.preview_height)
            self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.preview_label.setStyleSheet("""
                QLabel {
                    background-color: #f0f0f0;
                    border: 2px solid #ccc;
                    border-radius: 5px;
                }
            """)
            self.preview_label.setText("No image to display")

            # Status label
            self.status_label = QLabel("No project loaded")
            self.status_label.setAlignment(Qt.AlignCenter)
            self.status_label.setStyleSheet("font-style: italic; color: #666;")

            # Add to layout
            layout.addWidget(self.preview_label, stretch=1)
            layout.addWidget(self.status_label)

            self.preview_group.setLayout(layout)

        except Exception as e:
            self.show_error("Setup Error", f"Failed to setup preview UI: {str(e)}")

    def create_styled_button(self, text, color, icon_name):
        """Helper to create consistently styled buttons"""
        btn = QPushButton(text)
        btn.setIcon(QIcon.fromTheme(icon_name))
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                min-width: 100px;
            }}
            QPushButton:hover {{
                background-color: {self.darken_color(color)};
            }}
            QPushButton:disabled {{
                background-color: #cccccc;
                color: #666666;
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
                self.start_webcam()
        except Exception as e:
            self.show_error("Source Error", f"Failed to switch source: {str(e)}")
            self.source_combo.setCurrentIndex(0)

    def start_webcam(self):
        """Initialize and start webcam capture"""
        try:
            if self.webcam_active:
                return

            self.cap = cv2.VideoCapture(0)
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
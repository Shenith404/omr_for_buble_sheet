import cv2
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QComboBox, QFileDialog
)
from PySide6.QtGui import QPixmap, QImage, QCursor
from PySide6.QtCore import Qt, Signal, QEvent, QTimer, QThread, QObject
import utils
import model



class ClickableLabel(QLabel):
    clicked = Signal()
    doubleClicked = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)
        
    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.doubleClicked.emit()
        super().mouseDoubleClickEvent(event)

# ====================== Worker Thread Implementation ======================
class OMRProcessor(QObject):
    finished = Signal(object, object, object)  # answers, processed_img, debug_imgs
    error = Signal(str)
    progress = Signal(str)
    cancelled = Signal()

    def __init__(self, image, width, height):
        super().__init__()
        self.image = image
        self.width = width
        self.height = height
        self._is_running = True

    def process_image(self):
        try:
            # Step 1: Preprocessing
            self.progress.emit("Preprocessing image...")
            img = cv2.resize(self.image.copy(), (self.width, self.height))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 1)
            edges = cv2.Canny(blur, 10, 50)
            
            if not self._is_running:
                self.cancelled.emit()
                return

            # Step 2: Contour Detection
            self.progress.emit("Detecting contours...")
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            rects = utils.rectContour(contours)
            
            if not rects:
                self.error.emit("No answer sheet detected")
                return

            # Step 3: Perspective Transform
            self.progress.emit("Correcting perspective...")
            biggest = utils.getCornerPoints(rects[0])
            if biggest.size == 0:
                self.error.emit("Could not detect corners")
                return
                
            biggest = utils.reorder(biggest)
            pts1 = np.float32(biggest)
            pts2 = np.float32([[0, 0], [self.width, 0], [0, self.height], [self.width, self.height]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            warped = cv2.warpPerspective(img, matrix, (self.width, self.height))

            # Step 4: Thresholding
            self.progress.emit("Applying threshold...")
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(warped_gray, 170, 255, cv2.THRESH_BINARY_INV)[1]

            # Step 5: Answer Detection
            self.progress.emit("Detecting answers...")
            boxes = utils.verticalSplitBoxes(thresh)
            answers = []
            
            for i, box in enumerate(boxes):
                if not self._is_running:
                    self.cancelled.emit()
                    return
                    
                answer_blocks = utils.getAnswerBlocks(box)[2:6]  # Skip first 2 and last 1
                marked = []
                
                for j, block in enumerate(answer_blocks):
                    if cv2.countNonZero(block) > 0:
                        label, _ = model.classify_bubble(block)
                        if label in ["Crossed_Bubble", "Cross_Removed_Bubble"]:
                            marked.append(j+2)
                
                answers.append(marked[0] if len(marked) == 1 else -1)

            # Step 6: Generate Results
            self.progress.emit("Generating results...")
            result_img = utils.showAnswers(warped, answers)
            drawing = np.zeros_like(warped)
            drawing = utils.showAnswers(drawing, answers)
            
            inv_matrix = cv2.getPerspectiveTransform(pts2, pts1)
            inv_drawing = cv2.warpPerspective(drawing, inv_matrix, (img.shape[1], img.shape[0]))
            final_img = cv2.addWeighted(img, 1, inv_drawing, 1, 0)

            # Debug images
            blank = np.zeros_like(img)
            debug_imgs = [
                [img.copy(), warped, thresh, blank],
                [result_img, drawing, final_img, blank]
            ]

            self.finished.emit(answers, final_img, debug_imgs)

        except Exception as e:
            self.error.emit(f"Processing error: {str(e)}")
        finally:
            self._is_running = False

    def stop(self):
        self._is_running = False

# ====================== Main Processing Tab ======================
class ProcessingTab(QWidget):
    processing_complete = Signal(object, object, object)
    processing_cancelled = Signal()

    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setup_variables()
        self.setup_connections()

    def setup_variables(self):
        self.widthImg = 1025
        self.heightImg = 760
        self.webCamFeed = False
        self.cap = None
        self.image = None
        self.processed_image = None
        self.threshold_image = None
        self.is_maximized = False
        self.zoom_factor = 1.0
        self.pan_offset = [0, 0]
        self.processing = False
        
        # Threading
        self.worker_thread = None
        self.omr_processor = None
        self.webcam_timer = QTimer()
        self.webcam_timer.setInterval(30)  # 30ms for ~33fps

    def setup_ui(self):
        self.setup_control_panel()
        self.setup_preview_area()
        self.setup_maximized_view()
        self.setup_main_layout()

    def setup_control_panel(self):
        self.control_group = QGroupBox("OMR Processing Controls")
        layout = QHBoxLayout()
        
        self.btn_load = QPushButton("📁 Load Image")
        self.btn_capture = QPushButton("📸 Capture")
        self.btn_cancel = QPushButton("❌ Cancel")
        self.source_combo = QComboBox()
        self.btn_process = QPushButton("🔍 Scan OMR Sheet")
        
        # Styling
        self.style_buttons()
        
        # Initial visibility
        self.btn_capture.hide()
        self.btn_cancel.hide()
        
        # Add to layout
        layout.addWidget(self.btn_load)
        layout.addWidget(self.btn_capture)
        layout.addWidget(self.btn_cancel)
        layout.addWidget(self.source_combo)
        layout.addWidget(self.btn_process)
        self.control_group.setLayout(layout)

    def style_buttons(self):
        self.btn_load.setStyleSheet("font-weight: bold;")
        self.btn_capture.setStyleSheet("""
            font-weight: bold; 
            background-color: #4CAF50; 
            color: white;
        """)
        self.btn_cancel.setStyleSheet("""
            font-weight: bold; 
            background-color: #f44336; 
            color: white;
        """)
        self.btn_process.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 5px;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)

    def setup_preview_area(self):
        self.preview_group = QGroupBox("Image Preview (Double-click to maximize)")
        layout = QHBoxLayout()
        
        # Create preview labels
        self.original_label = ClickableLabel()
        self.threshold_label = ClickableLabel()
        self.processed_label = ClickableLabel()
        
        # Configure labels
        for label in [self.original_label, self.threshold_label, self.processed_label]:
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("""
                background-color: #f0f0f0;
                border: 2px solid #cccccc;
            """)
            label.setMinimumSize(400, 300)
        
        # Connect signals
        self.original_label.doubleClicked.connect(lambda: self.toggle_maximize(self.original_label))
        self.threshold_label.doubleClicked.connect(lambda: self.toggle_maximize(self.threshold_label))
        self.processed_label.doubleClicked.connect(lambda: self.toggle_maximize(self.processed_label))
        
        # Add to layout
        layout.addWidget(self.original_label)
        layout.addWidget(self.threshold_label)
        layout.addWidget(self.processed_label)
        self.preview_group.setLayout(layout)

    def setup_maximized_view(self):
        self.maximized_view = ClickableLabel()
        self.maximized_view.setAlignment(Qt.AlignCenter)
        self.maximized_view.setStyleSheet("background-color: #000000;")
        self.maximized_view.hide()
        
        self.close_maximized_btn = QPushButton("✕ Close (ESC)")
        self.close_maximized_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 180);
                color: #333333;
                border: none;
                padding: 5px 10px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: rgba(255, 0, 0, 180);
                color: white;
            }
        """)
        self.close_maximized_btn.setFixedSize(100, 30)
        self.close_maximized_btn.hide()
        
        # Make them top-level windows
        self.maximized_view.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.close_maximized_btn.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)

    def setup_main_layout(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Status label
        self.status_label = QLabel("Ready to process OMR sheet")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-style: italic; color: #666666;")
        
        # Add widgets
        main_layout.addWidget(self.control_group)
        main_layout.addWidget(self.preview_group)
        main_layout.addWidget(self.status_label)
        
        self.setLayout(main_layout)

    def setup_connections(self):
        # Button connections
        self.btn_load.clicked.connect(self.load_image)
        self.btn_capture.clicked.connect(self.capture_webcam_image)
        self.btn_cancel.clicked.connect(self.cancel_operation)
        self.btn_process.clicked.connect(self.process_omr)
        self.close_maximized_btn.clicked.connect(self.restore_previews)
        
        # Combo box
        self.source_combo.addItems(["Image File", "Webcam"])
        self.source_combo.currentIndexChanged.connect(self.toggle_webcam)
        
        # Timer
        self.webcam_timer.timeout.connect(self.update_webcam_preview)
        
        # Maximized view
        self.maximized_view.doubleClicked.connect(self.restore_previews)
        self.maximized_view.installEventFilter(self)

    # ====================== Core Functionality ======================
    def toggle_webcam(self, index):
        self.webCamFeed = (index == 1)
        if self.webCamFeed:
            self.setup_webcam()
        else:
            self.cleanup_webcam()

    def setup_webcam(self):
        self.btn_load.hide()
        self.btn_capture.show()
        self.btn_cancel.show()
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.webcam_timer.start()
            self.status_label.setText("Webcam active - position your OMR sheet")
        else:
            self.status_label.setText("Error: Could not open webcam")
            self.source_combo.setCurrentIndex(0)

    def cleanup_webcam(self):
        self.btn_load.show()
        self.btn_capture.hide()
        self.btn_cancel.hide()
        if self.cap:
            self.webcam_timer.stop()
            self.cap.release()
            self.cap = None
        self.status_label.setText("Ready to process OMR sheet")

    def update_webcam_preview(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.show_image(frame, self.original_label)

    def capture_webcam_image(self):
        if self.cap and self.cap.isOpened():
            ret, self.image = self.cap.read()
            if ret:
                self.webcam_timer.stop()
                self.btn_process.setEnabled(True)
                self.btn_capture.hide()
                self.btn_cancel.hide()
                self.status_label.setText("Webcam image captured - ready to process")
                self.show_image(self.image, self.original_label)

    def load_image(self):
        if self.webCamFeed:
            return
            
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open OMR Sheet", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_path:
            self.image = cv2.imread(file_path)
            if self.image is not None:
                self.show_image(self.image, self.original_label)
                self.btn_process.setEnabled(True)
                self.status_label.setText(f"Loaded: {file_path.split('/')[-1]}")
            else:
                self.status_label.setText("Error: Could not load image")

    def process_omr(self):
        if self.image is None:
            self.status_label.setText("Error: No image loaded")
            return
            
        # Setup processing thread
        self.processing = True
        self.set_ui_processing_state(True)
        
        self.worker_thread = QThread()
        self.omr_processor = OMRProcessor(
            self.image.copy(), 
            self.widthImg, 
            self.heightImg
        )
        
        # Move worker to thread
        self.omr_processor.moveToThread(self.worker_thread)
        
        # Connect signals
        self.worker_thread.started.connect(self.omr_processor.process_image)
        self.omr_processor.finished.connect(self.on_processing_complete)
        self.omr_processor.error.connect(self.on_processing_error)
        self.omr_processor.progress.connect(self.status_label.setText)
        self.omr_processor.cancelled.connect(self.on_processing_cancelled)
        
        # Cleanup connections
        for signal in [self.omr_processor.finished, 
                      self.omr_processor.error, 
                      self.omr_processor.cancelled]:
            signal.connect(self.worker_thread.quit)
            signal.connect(self.omr_processor.deleteLater)
        
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        
        # Start processing
        self.worker_thread.start()

    def on_processing_complete(self, answers, processed_img, debug_imgs):
        self.processing = False
        self.processed_image = processed_img
        self.show_results(processed_img, debug_imgs)
        self.processing_complete.emit(answers, processed_img, debug_imgs)
        self.status_label.setText("Processing complete!")
        self.set_ui_processing_state(False)

    def on_processing_error(self, error_msg):
        self.processing = False
        self.status_label.setText(error_msg)
        self.set_ui_processing_state(False)

    def on_processing_cancelled(self):
        self.processing = False
        self.status_label.setText("Processing cancelled")
        self.set_ui_processing_state(False)
        self.processing_cancelled.emit()

    def show_results(self, processed_img, debug_imgs):
        self.show_image(processed_img, self.processed_label)
        if debug_imgs and len(debug_imgs) > 0 and debug_imgs[0][2] is not None:
            self.show_image(debug_imgs[0][2], self.threshold_label)

    def set_ui_processing_state(self, processing):
        self.btn_process.setEnabled(not processing)
        self.btn_load.setEnabled(not processing)
        self.source_combo.setEnabled(not processing)
        self.btn_cancel.setVisible(processing)

    def cancel_operation(self):
        if self.processing:
            if self.omr_processor:
                self.omr_processor.stop()
            if self.worker_thread and self.worker_thread.isRunning():
                self.worker_thread.quit()
                self.worker_thread.wait()
        elif self.webCamFeed:
            self.cleanup_webcam()
            self.source_combo.setCurrentIndex(0)

    # ====================== Image Display Methods ======================
    def show_image(self, cv_img, label):
        if cv_img is None:
            return
            
        # Convert OpenCV image to QImage
        if len(cv_img.shape) == 2:  # Grayscale
            q_img = QImage(
                cv_img.data, cv_img.shape[1], cv_img.shape[0], 
                cv_img.shape[1], QImage.Format_Grayscale8
            )
        else:  # Color
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            q_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        
        # Create and set pixmap
        pixmap = QPixmap.fromImage(q_img).scaled(
            label.width(), label.height(), 
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        label.setPixmap(pixmap)
        
        # Store full resolution for maximized view
        if label == self.original_label:
            self.original_pixmap = QPixmap.fromImage(q_img)
        elif label == self.processed_label:
            self.processed_pixmap = QPixmap.fromImage(q_img)

    def toggle_maximize(self, label):
        if self.is_maximized:
            self.restore_previews()
            return
            
        # Get the appropriate pixmap
        if label == self.original_label and hasattr(self, 'original_pixmap'):
            pixmap = self.original_pixmap
        elif label == self.processed_label and hasattr(self, 'processed_pixmap'):
            pixmap = self.processed_pixmap
        else:
            return
            
        # Setup maximized view
        screen = self.screen().availableGeometry()
        self.maximized_view.setGeometry(screen)
        self.maximized_view.setPixmap(pixmap.scaled(
            screen.width(), screen.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        
        # Position close button
        self.close_maximized_btn.move(screen.width() - 110, 10)
        
        # Show/hide elements
        self.maximized_view.show()
        self.close_maximized_btn.show()
        self.preview_group.hide()
        self.is_maximized = True
        self.current_maximized_label = label
        
        # Reset view state
        self.zoom_factor = 1.0
        self.pan_offset = [0, 0]

    def restore_previews(self):
        if self.is_maximized:
            self.maximized_view.hide()
            self.close_maximized_btn.hide()
            self.preview_group.show()
            self.is_maximized = False
            self.maximized_view.unsetCursor()

    def eventFilter(self, obj, event):
        if obj == self.maximized_view and self.is_maximized:
            if event.type() == QEvent.MouseButtonPress:
                if event.button() == Qt.LeftButton:
                    self.drag_start = event.pos()
                    self.maximized_view.setCursor(Qt.ClosedHandCursor)
                    return True
                    
            elif event.type() == QEvent.MouseMove:
                if hasattr(self, 'drag_start'):
                    delta = event.pos() - self.drag_start
                    self.pan_offset[0] += delta.x()
                    self.pan_offset[1] += delta.y()
                    self.drag_start = event.pos()
                    self.update_maximized_view()
                    return True
                    
            elif event.type() == QEvent.MouseButtonRelease:
                if hasattr(self, 'drag_start'):
                    del self.drag_start
                    self.maximized_view.setCursor(Qt.ArrowCursor)
                    return True
                    
            elif event.type() == QEvent.Wheel:
                degrees = event.angleDelta().y() / 8
                steps = degrees / 15
                self.zoom_factor *= 1.1 if steps > 0 else 0.9
                self.zoom_factor = max(1.0, self.zoom_factor)
                if self.zoom_factor == 1.0:
                    self.pan_offset = [0, 0]
                self.update_maximized_view()
                return True
                
        return super().eventFilter(obj, event)

    def update_maximized_view(self):
        if not hasattr(self, 'current_maximized_label'):
            return
            
        pixmap = getattr(self, f"{self.current_maximized_label.objectName().replace('_label', '')}_pixmap")
        if not pixmap:
            return
            
        screen = self.screen().availableGeometry()
        scaled_w = int(screen.width() * self.zoom_factor)
        scaled_h = int(screen.height() * self.zoom_factor)
        
        transformed = pixmap.scaled(
            scaled_w, scaled_h,
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        
        x_offset = max(0, min(self.pan_offset[0], scaled_w - screen.width()))
        y_offset = max(0, min(self.pan_offset[1], scaled_h - screen.height()))
        
        if x_offset > 0 or y_offset > 0:
            cropped = transformed.copy(
                x_offset, y_offset,
                screen.width(), screen.height()
            )
            self.maximized_view.setPixmap(cropped)
        else:
            self.maximized_view.setPixmap(transformed)

    def closeEvent(self, event):
        # Cleanup resources
        if hasattr(self, 'worker_thread') and self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait()
        
        if self.cap:
            self.cap.release()
            
        if self.webcam_timer.isActive():
            self.webcam_timer.stop()
            
        event.accept()
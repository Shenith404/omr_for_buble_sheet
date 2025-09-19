"""
UI Builder for Verify Tab
Handles all UI setup and styling for the verify tab
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QSizePolicy, QFileDialog,
    QComboBox, QSpacerItem, QMessageBox, QSpinBox,
    QGridLayout, QScrollArea, QFrame, QToolTip,
    QLineEdit, QListWidget, QListWidgetItem
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt, Signal


class VerifyUIBuilder:
    """Handles UI creation and styling for the Verify Tab"""
    
    @staticmethod
    def get_main_stylesheet():
        """Return the main stylesheet for the verify tab"""
        return """
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
            QComboBox {
                background: #2a2a2a;
                border: 1px solid #404040;
                border-radius: 6px;
                color: #e8e8e8;
                font-size: 13px;
                padding: 8px 12px;
                selection-background-color: #0078d4;
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
        """

    @staticmethod
    def create_project_status_group():
        """Create the project status group"""
        project_group = QGroupBox("Project Status")
        project_layout = QHBoxLayout()
        project_layout.setSpacing(15)
        
        lbl_project = QLabel("No project loaded")
        lbl_project.setStyleSheet("""
            QLabel {
                background: #2a2a2a;
                border: 1px solid #404040;
                border-radius: 6px;
                padding: 10px 12px;
                color: #cccccc;
                font-style: italic;
                font-weight: 500;
            }
        """)
        
        project_layout.addWidget(lbl_project, 1)
        project_group.setLayout(project_layout)
        
        return project_group, lbl_project

    @staticmethod
    def create_search_and_filter_widgets():
        """Create search input and filter combo widgets"""
        # Search input
        search_input = QLineEdit()
        search_input.setPlaceholderText("🔍 Search images...")
        search_input.setStyleSheet("""
            QLineEdit {
                background: #2a2a2a;
                border: 1px solid #404040;
                border-radius: 6px;
                color: #e8e8e8;
                font-size: 13px;
                padding: 8px 12px;
                min-height: 20px;
            }
            QLineEdit:focus {
                border-color: #0078d4;
                background: #323232;
            }
            QLineEdit::placeholder {
                color: #888888;
            }
        """)
        
        # Filter combo
        filter_combo = QComboBox()
        filter_combo.addItems(["Reviewed Images", "Verified Images", "Unverified Images"])
        filter_combo.setStyleSheet("""
            QComboBox {
                background: #2a2a2a;
                border: 1px solid #404040;
                border-radius: 6px;
                color: #e8e8e8;
                font-size: 13px;
                padding: 8px 12px;
                min-width: 120px;
                min-height: 20px;
            }
            QComboBox:focus {
                border-color: #0078d4;
                background: #323232;
            }
        """)
        
        return search_input, filter_combo

    @staticmethod
    def create_navigation_buttons():
        """Create navigation buttons"""
        btn_prev = QPushButton("◀ Previous")
        btn_prev.setStyleSheet("""
            QPushButton {
                background: linear-gradient(180deg, #6c757d 0%, #495057 100%);
                border: 1px solid #495057;
                min-width: 100px;
            }
            QPushButton:hover {
                background: linear-gradient(180deg, #868e96 0%, #6c757d 100%);
            }
        """)
        
        btn_next = QPushButton("Next ▶")
        btn_next.setStyleSheet("""
            QPushButton {
                background: linear-gradient(180deg, #6c757d 0%, #495057 100%);
                border: 1px solid #495057;
                min-width: 100px;
            }
            QPushButton:hover {
                background: linear-gradient(180deg, #868e96 0%, #6c757d 100%);
            }
        """)
        
        lbl_image_info = QLabel("0/0 images loaded")
        lbl_image_info.setStyleSheet("""
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
        lbl_image_info.setAlignment(Qt.AlignCenter)
        
        return btn_prev, btn_next, lbl_image_info

    @staticmethod
    def create_image_display_label():
        """Create the main image display label"""
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setStyleSheet("""
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
        image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        image_label.setText(" No reviewed images to verify\n\nOpen a project from the Project tab to begin")
        
        return image_label

    @staticmethod
    def create_results_info_group():
        """Create the results information group"""
        info_group = QGroupBox("Results Information")
        info_layout = QVBoxLayout()
        
        lbl_results_info = QLabel("No results available")
        lbl_results_info.setWordWrap(True)
        lbl_results_info.setStyleSheet("""
            QLabel {
                background: #2a2a2a;
                border: 1px solid #404040;
                border-radius: 6px;
                padding: 12px;
                color: #e8e8e8;
                font-size: 13px;
                line-height: 1.4;
            }
        """)
        info_layout.addWidget(lbl_results_info)
        info_group.setLayout(info_layout)
        
        return info_group, lbl_results_info

    @staticmethod
    def create_download_results_button():
        """Create the download results button"""
        btn_download_results = QPushButton(" Download Results")
        btn_download_results.setStyleSheet("""
            QPushButton {
                background: linear-gradient(180deg, #28a745 0%, #1e7e34 100%);
                border: 1px solid #1e7e34;
                font-weight: 600;
                padding: 12px 16px;
                border-radius: 8px;
                font-size: 14px;
                min-height: 20px;
            }
            QPushButton:hover {
                background: linear-gradient(180deg, #34d058 0%, #28a745 100%);
            }
            QPushButton:disabled {
                background: #3a3a3a;
                border-color: #525252;
                color: #7a7a7a;
            }
        """)
        return btn_download_results

    @staticmethod
    def create_images_list_widget():
        """Create the images list widget"""
        images_list = QListWidget()
        images_list.setStyleSheet("""
            QListWidget {
                background: #2a2a2a;
                border: 1px solid #404040;
                border-radius: 6px;
                color: #e8e8e8;
                font-size: 12px;
                padding: 5px;
                selection-background-color: #0078d4;
                selection-color: #ffffff;
                outline: none;
            }
            QListWidget::item {
                background: transparent;
                border: none;
                border-radius: 4px;
                padding: 8px 10px;
                margin: 2px 0px;
            }
            QListWidget::item:hover {
                background: #404040;
            }
            QListWidget::item:selected {
                background: #0078d4;
                color: #ffffff;
            }
            QListWidget::item:selected:hover {
                background: #1084d8;
            }
        """)
        images_list.setMaximumHeight(200)
        
        return images_list

    @staticmethod
    def create_file_rename_widgets():
        """Create file rename widgets"""
        # Current filename display
        current_filename_label = QLabel("Current: No file selected")
        current_filename_label.setStyleSheet("""
            QLabel {
                background: #2a2a2a;
                border: 1px solid #404040;
                border-radius: 4px;
                padding: 6px 8px;
                color: #cccccc;
                font-size: 11px;
                font-style: italic;
            }
        """)
        
        # New filename input
        filename_input = QLineEdit()
        filename_input.setPlaceholderText("Enter new filename...")
        filename_input.setStyleSheet("""
            QLineEdit {
                background: #2a2a2a;
                border: 1px solid #404040;
                border-radius: 4px;
                color: #e8e8e8;
                font-size: 11px;
                padding: 6px 8px;
                min-height: 12px;
            }
            QLineEdit:focus {
                border-color: #0078d4;
                background: #323232;
            }
            QLineEdit::placeholder {
                color: #888888;
            }
        """)
        
        file_extension_label = QLabel(".jpg")
        file_extension_label.setStyleSheet("""
            QLabel {
                background: #3a3a3a;
                border: 1px solid #505050;
                border-radius: 4px;
                padding: 6px 8px;
                color: #cccccc;
                font-size: 11px;
                font-weight: 500;
                min-width: 35px;
            }
        """)
        
        btn_rename_file = QPushButton("Rename")
        btn_rename_file.setStyleSheet("""
            QPushButton {
                background: linear-gradient(180deg, #28a745 0%, #1e7e34 100%);
                border: 1px solid #1e7e34;
                font-weight: 600;
                padding: 6px 12px;
                border-radius: 4px;
                font-size: 11px;
                color: #ffffff;
                min-width: 50px;
                max-height: 26px;
            }
            QPushButton:hover {
                background: linear-gradient(180deg, #34ce57 0%, #28a745 100%);
            }
            QPushButton:pressed {
                background: linear-gradient(180deg, #1e7e34 0%, #155724 100%);
            }
            QPushButton:disabled {
                background: #3a3a3a;
                border-color: #525252;
                color: #7a7a7a;
            }
        """)
        
        return current_filename_label, filename_input, file_extension_label, btn_rename_file

    @staticmethod
    def create_question_grid(question_buttons_list):
        """Create the question selection grid"""
        # Question Grid Container with Scroll
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setMinimumHeight(120)
        scroll_area.setMaximumHeight(200)
        scroll_area.setStyleSheet("""
            QScrollArea {
                background: #2a2a2a;
                border: 1px solid #404040;
                border-radius: 8px;
            }
            QScrollBar:vertical {
                background: #1e1e1e;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #505050;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #606060;
            }
        """)
        
        # Question Grid Widget
        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setSpacing(2)
        grid_layout.setContentsMargins(6, 6, 6, 6)
        
        # Create question buttons (1-50)
        for i in range(50):
            question_num = i + 1
            btn = QPushButton(str(question_num))
            btn.setFixedSize(22, 22)
            btn.setCheckable(False)
            
            # Style for question buttons
            btn.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #3a3a3a, stop:1 #2a2a2a);
                    border: 1px solid #505050;
                    border-radius: 4px;
                    color: #e8e8e8;
                    font-size: 11px;
                    font-weight: 600;
                    padding: 0px;
                    margin: 0px;
                    min-width: 20px;
                    min-height: 20px;
                    max-width: 22px;
                    max-height: 22px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #4a4a4a, stop:1 #3a3a3a);
                    border: 2px solid #0078d4;
                    color: #ffffff;
                }
                QPushButton:checked {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #0078d4, stop:1 #005a9e);
                    border: 2px solid #0078d4;
                    color: #ffffff;
                    font-weight: 700;
                }
                QPushButton:checked:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #1084d8, stop:1 #0066b2);
                    border: 2px solid #1084d8;
                }
                QPushButton:pressed {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #005a9e, stop:1 #004578);
                }
            """)
            
            # Add tooltip
            btn.setToolTip(f"Question {question_num}\nClick to select")
            
            question_buttons_list.append(btn)
            grid_layout.addWidget(btn, i // 10, i % 10)  # 10 columns per row
        
        scroll_area.setWidget(grid_widget)
        return scroll_area

    @staticmethod
    def create_selected_question_label():
        """Create the selected question label"""
        selected_question_label = QLabel("Selected: Question 1")
        selected_question_label.setStyleSheet("""
            QLabel {
                background: #2a2a2a;
                border: 1px solid #404040;
                border-radius: 6px;
                padding: 8px 12px;
                color: #ffffff;
                font-weight: 600;
                font-size: 13px;
            }
        """)
        return selected_question_label

    @staticmethod
    def create_answer_buttons():
        """Create answer selection buttons"""
        answer_buttons = []
        answer_labels = ['A', 'B', 'C', 'D']
        
        for i, label in enumerate(answer_labels):
            btn = QPushButton(f"{i+1} ({label})")
            btn.setCheckable(True)
            btn.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #3a3a3a, stop:1 #2a2a2a);
                    border: 1px solid #505050;
                    border-radius: 8px;
                    color: #e8e8e8;
                    font-size: 12px;
                    font-weight: 500;
                    padding: 8px 4px;
                    min-width: 55px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #4a4a4a, stop:1 #3a3a3a);
                    border: 2px solid #0078d4;
                    color: #ffffff;
                }
                QPushButton:checked {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #28a745, stop:1 #1e7e34);
                    border: 2px solid #28a745;
                    color: #ffffff;
                    font-weight: 600;
                }
                QPushButton:checked:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #34ce57, stop:1 #28a745);
                }
                QPushButton:pressed {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #1e7e34, stop:1 #155724);
                }
            """)
            answer_buttons.append(btn)
        
        return answer_buttons

    @staticmethod
    def create_action_buttons():
        """Create action buttons (Change Answer and Verified by Second Examiner)"""
        # Change Answer button
        btn_change_answer = QPushButton("🔄 Update Answer")
        btn_change_answer.setStyleSheet("""
            QPushButton {
                background: linear-gradient(180deg, #007bff 0%, #0056b3 100%);
                border: 1px solid #0056b3;
                font-weight: 600;
                padding: 10px 16px;
                border-radius: 8px;
                font-size: 13px;
                color: #ffffff;
                min-height: 15px;
            }
            QPushButton:hover {
                background: linear-gradient(180deg, #1a8cff 0%, #007bff 100%);
            }
            QPushButton:pressed {
                background: linear-gradient(180deg, #0056b3 0%, #004085 100%);
            }
            QPushButton:disabled {
                background: #3a3a3a;
                border-color: #525252;
                color: #7a7a7a;
            }
        """)
        
        # Verified by Second Examiner button (Changed from "Mark as Reviewed")
        btn_mark_reviewed = QPushButton("✅ Verified by Second Examiner")
        btn_mark_reviewed.setStyleSheet("""
            QPushButton {
                background: linear-gradient(180deg, #6f42c1 0%, #5a2d8c 100%);
                border: 1px solid #5a2d8c;
                font-weight: 600;
                padding: 12px 16px;
                border-radius: 8px;
                font-size: 14px;
                min-height: 20px;
            }
            QPushButton:hover {
                background: linear-gradient(180deg, #8b5cf6 0%, #6f42c1 100%);
            }
            QPushButton:disabled {
                background: #3a3a3a;
                border-color: #525252;
                color: #7a7a7a;
            }
        """)
        
        return btn_change_answer, btn_mark_reviewed
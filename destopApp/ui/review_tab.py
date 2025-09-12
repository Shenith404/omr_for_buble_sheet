import os
import json
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QSizePolicy, QFileDialog,
    QComboBox, QSpacerItem, QMessageBox, QSpinBox,
    QGridLayout, QScrollArea, QFrame, QToolTip
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt, Signal
import db  
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
        self.setStyleSheet("""
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
        """)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Left Panel (70%)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(15)
        
        # Project Status Group
        project_group = QGroupBox("Project Status")
        project_layout = QHBoxLayout()
        project_layout.setSpacing(15)
        
        self.lbl_project = QLabel("No project loaded")
        self.lbl_project.setStyleSheet("""
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
        
        project_layout.addWidget(self.lbl_project, 1)
        project_group.setLayout(project_layout)
        
        # Image Navigation Group
        nav_group = QGroupBox("Image Navigation")
        nav_layout = QVBoxLayout()
        nav_layout.setSpacing(15)
        
        # Search and Filter Row
        search_layout = QHBoxLayout()
        search_layout.setSpacing(10)
        
        # Search input
        from PySide6.QtWidgets import QLineEdit
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔍 Search images...")
        self.search_input.setStyleSheet("""
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
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All Images", "Reviewed Only", "Unreviewed Only"])
        self.filter_combo.setStyleSheet("""
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
        
        search_layout.addWidget(self.search_input, 2)
        search_layout.addWidget(self.filter_combo, 1)
        
        # Navigation buttons row
        nav_buttons_layout = QHBoxLayout()
        nav_buttons_layout.setSpacing(15)
        
        self.btn_prev = QPushButton("◀ Previous")
        self.btn_prev.setStyleSheet("""
            QPushButton {
                background: linear-gradient(180deg, #6c757d 0%, #495057 100%);
                border: 1px solid #495057;
                min-width: 100px;
            }
            QPushButton:hover {
                background: linear-gradient(180deg, #868e96 0%, #6c757d 100%);
            }
        """)
        
        self.btn_next = QPushButton("Next ▶")
        self.btn_next.setStyleSheet("""
            QPushButton {
                background: linear-gradient(180deg, #6c757d 0%, #495057 100%);
                border: 1px solid #495057;
                min-width: 100px;
            }
            QPushButton:hover {
                background: linear-gradient(180deg, #868e96 0%, #6c757d 100%);
            }
        """)
        
        self.lbl_image_info = QLabel("0/0 images loaded")
        self.lbl_image_info.setStyleSheet("""
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
        self.lbl_image_info.setAlignment(Qt.AlignCenter)
        
        nav_buttons_layout.addWidget(self.btn_prev)
        nav_buttons_layout.addWidget(self.lbl_image_info)
        nav_buttons_layout.addWidget(self.btn_next)
        
        nav_layout.addLayout(search_layout)
        nav_layout.addLayout(nav_buttons_layout)
        nav_group.setLayout(nav_layout)
        
        # Image Display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
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
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setText(" No results to review\n\nOpen a project from the Project tab to begin")
        
        # Results Info Group
        info_group = QGroupBox("Results Information")
        info_layout = QVBoxLayout()
        
        self.lbl_results_info = QLabel("No results available")
        self.lbl_results_info.setWordWrap(True)
        self.lbl_results_info.setStyleSheet("""
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
            QWidget {
                border-left: 2px solid #404040;
                padding-left: 15px;
            }
        """)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(15, 0, 0, 0)
        right_layout.setSpacing(20)
        
        # Download Results Button (added at top of right panel)
        self.btn_download_results = QPushButton(" Download Results")
        self.btn_download_results.setStyleSheet("""
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
        right_layout.addWidget(self.btn_download_results)
        
        # Image List Group
        from PySide6.QtWidgets import QListWidget, QListWidgetItem
        images_group = QGroupBox("Images")
        images_layout = QVBoxLayout(images_group)
        images_layout.setSpacing(10)
        
        self.images_list = QListWidget()
        self.images_list.setStyleSheet("""
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
        self.images_list.setMaximumHeight(200)
        
        images_layout.addWidget(self.images_list)
        right_layout.addWidget(images_group)
        
        # Question Selection Group (Enhanced UI)
        question_group = QGroupBox("Change Detected Answers")
        question_layout = QVBoxLayout(question_group)
        question_layout.setSpacing(12)
        
        # Question Selection Info
        selection_info = QLabel("Click a question number to select:")
        selection_info.setStyleSheet("QLabel { color: #b8b8b8; font-size: 12px; margin-bottom: 5px; }")
        question_layout.addWidget(selection_info)
        
        # Keyboard shortcuts help
        shortcuts_help = QLabel("🎯 Shortcuts: ↑↓ Navigate | 1-4 Select Answer | Enter Apply | Space Review")
        shortcuts_help.setStyleSheet("""
            QLabel { 
                color: #888888; 
                font-size: 10px; 
                background: #1a1a1a;
                border: 1px solid #333333;
                border-radius: 4px;
                padding: 4px 8px;
                margin-bottom: 8px;
            }
        """)
        shortcuts_help.setWordWrap(True)
        question_layout.addWidget(shortcuts_help)
        
        # Question Grid Container with Scroll (responsive height)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setMinimumHeight(120)  # Minimum height for usability
        scroll_area.setMaximumHeight(200)  # Maximum height to prevent taking too much space
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
        grid_layout.setSpacing(2)  # Further reduced spacing for smaller buttons
        grid_layout.setContentsMargins(6, 6, 6, 6)
        
        # Create question buttons (1-50)
        self.question_buttons = []
        for i in range(50):
            question_num = i + 1
            btn = QPushButton(str(question_num))
            btn.setFixedSize(22, 22)  # Even smaller size
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, q=question_num: self.select_question(q))
            
            # Style for question buttons with zero padding and smaller size
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
            
            self.question_buttons.append(btn)
            grid_layout.addWidget(btn, i // 10, i % 10)  # 10 columns per row
        
        # Set first question as selected by default
        if self.question_buttons:
            self.question_buttons[0].setChecked(True)
        
        scroll_area.setWidget(grid_widget)
        question_layout.addWidget(scroll_area)
        
        # Selected Question Info
        self.selected_question_label = QLabel("Selected: Question 1")
        self.selected_question_label.setStyleSheet("""
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
        question_layout.addWidget(self.selected_question_label)
        
        # Answer Section (Enhanced)
        answer_label = QLabel("New Answer:")
        answer_label.setStyleSheet("QLabel { color: #e8e8e8; font-weight: 500; margin-bottom: 3px; }")
        
        # Answer buttons instead of combo box for better UX
        answer_buttons_layout = QHBoxLayout()
        answer_buttons_layout.setSpacing(6)
        
        self.answer_buttons = []
        answer_labels = ['A', 'B', 'C', 'D']
        for i, label in enumerate(answer_labels):
            btn = QPushButton(f"{i+1} ({label})")
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, ans=i+1: self.select_answer(ans))
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
            self.answer_buttons.append(btn)
            answer_buttons_layout.addWidget(btn)
        
        # Set first answer as selected by default
        if self.answer_buttons:
            self.answer_buttons[0].setChecked(True)
        
        # Legacy combo box (hidden but kept for compatibility)
        self.answer_combo = QComboBox()
        self.answer_combo.addItems([str(i) for i in range(1, 5)])
        self.answer_combo.hide()  # Hide the old combo box
        
        # Change Answer button (Enhanced)
        self.btn_change_answer = QPushButton("🔄 Update Answer")
        self.btn_change_answer.setStyleSheet("""
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
        
        question_layout.addWidget(answer_label)
        question_layout.addLayout(answer_buttons_layout)
        question_layout.addWidget(self.btn_change_answer)
        
        # Add question group to right layout
        right_layout.addWidget(question_group)
        right_layout.addStretch(1)  # Add flexible space
        
        # Mark as Reviewed button at bottom
        self.btn_mark_reviewed = QPushButton("✅ Mark as Reviewed")
        self.btn_mark_reviewed.setStyleSheet("""
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
        right_layout.addWidget(self.btn_mark_reviewed)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel, 7)  # 70% width
        main_layout.addWidget(right_panel, 3)  # 30% width
        
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

    def select_question(self, question_num):
        """Handle question selection from grid"""
        self.selected_question = question_num
        
        # Update button states
        for i, btn in enumerate(self.question_buttons):
            btn.setChecked(i + 1 == question_num)
        
        # Update selected question label
        self.selected_question_label.setText(f"Selected: Question {question_num}")
        
        # Update question button appearance with current answer if available
        self.update_question_button_appearance(question_num)
    
    def select_answer(self, answer_num):
        """Handle answer selection from buttons"""
        # Update answer button states
        for i, btn in enumerate(self.answer_buttons):
            btn.setChecked(i + 1 == answer_num)
        
        # Update the hidden combo box for compatibility
        self.answer_combo.setCurrentText(str(answer_num))
    
    def update_question_button_appearance(self, question_num=None):
        """Update question button appearance to show current answers"""
        if not self.image_paths or self.current_index >= len(self.image_paths):
            return
        
        current_image = os.path.basename(self.image_paths[self.current_index])
        
        # Get current answers for this image from database
        try:
            if self.handler:
                sheet_data = self.handler.get_sheet(current_image)
                if sheet_data and 'detected_answers' in sheet_data:
                    answers = sheet_data['detected_answers']
                    for i, btn in enumerate(self.question_buttons):
                        q_num = i + 1
                        if i < len(answers):
                            answer = answers[i]  # 0-based index
                            
                            # Update button style and tooltip based on answer
                            if answer == -1:  # No answer detected
                                btn.setStyleSheet("""
                                    QPushButton {
                                        background: #4a4a4a;
                                        border: 1px solid #666666;
                                        border-radius: 4px;
                                        color: #cccccc;
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
                                        background: #5a5a5a;
                                        border-color: #0078d4;
                                    }
                                    QPushButton:checked {
                                        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #0078d4, stop:1 #005a9e);
                                        border: 2px solid #0078d4;
                                        color: #ffffff;
                                    }
                                """)
                                btn.setToolTip(f"Question {q_num}\nNo answer detected")
                            else:
                                # Convert 1-based answer to letter (2->A, 3->B, 4->C, 5->D)
                                answer_letter = chr(ord('A') + answer - 2) if answer >= 2 else '?'
                                btn.setStyleSheet("""
                                    QPushButton {
                                        background: #2d5a2d;
                                        border: 1px solid #4a8f4a;
                                        border-radius: 4px;
                                        color: #ffffff;
                                        font-size: 9px;
                                        font-weight: 600;
                                        padding: 0px;
                                        margin: 0px;
                                        min-width: 20px;
                                        min-height: 20px;
                                        max-width: 22px;
                                        max-height: 22px;
                                    }
                                    QPushButton:hover {
                                        background: #3a6a3a;
                                        border-color: #0078d4;
                                    }
                                    QPushButton:checked {
                                        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #0078d4, stop:1 #005a9e);
                                        border: 2px solid #0078d4;
                                        color: #ffffff;
                                    }
                                """)
                                btn.setToolTip(f"Question {q_num}\nDetected: {answer_letter}")
                                
                                # Show answer letter on button
                                btn.setText(f"{q_num}\n{answer_letter}")
                        else:
                            # Reset to default style
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
                                }
                            """)
                            btn.setText(str(q_num))
                            btn.setToolTip(f"Question {q_num}\nClick to select")
        except Exception as e:
            print(f"Error updating question buttons: {e}")
            # Reset all buttons to default on error
            for i, btn in enumerate(self.question_buttons):
                btn.setText(str(i + 1))
                btn.setToolTip(f"Question {i + 1}\nClick to select")

    def apply_filters(self):
        """Apply search and review status filters to image list"""
        if not self.image_paths:
            self.filtered_image_paths = []
            self.update_image_list()
            return
            
        search_text = self.search_input.text().lower()
        filter_type = self.filter_combo.currentText()
        
        # Start with all images
        filtered_paths = self.image_paths.copy()
        
        # Apply search filter
        if search_text:
            filtered_paths = [
                path for path in filtered_paths
                if search_text in os.path.basename(path).lower()
            ]
        
        # Apply review status filter
        if filter_type == "Reviewed Only":
            filtered_paths = [
                path for path in filtered_paths
                if self.is_image_reviewed(os.path.basename(path))
            ]
        elif filter_type == "Unreviewed Only":
            filtered_paths = [
                path for path in filtered_paths
                if not self.is_image_reviewed(os.path.basename(path))
            ]
        
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
        if not self.handler:
            return filename in self.reviewed_images
        
        sheet_data = self.handler.get_sheet(filename)
        if sheet_data:
            return sheet_data.get("reviewed", False)
        return filename in self.reviewed_images
    
    def update_image_list(self):
        """Update the image list widget with filtered results"""
        self.images_list.clear()
        
        for image_path in self.filtered_image_paths:
            filename = os.path.basename(image_path)
            is_reviewed = self.is_image_reviewed(filename)
            
            # Create list item with status indicator
            status_icon = "✅" if is_reviewed else "⏳"
            item_text = f"{status_icon} {filename}"
            
            from PySide6.QtWidgets import QListWidgetItem
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, image_path)  # Store full path in item data
            
            # Set different colors for reviewed/unreviewed
            if is_reviewed:
                item.setToolTip(f"Reviewed: {filename}")
            else:
                item.setToolTip(f"Unreviewed: {filename}")
            
            self.images_list.addItem(item)
        
        # Update image count info
        total_images = len(self.image_paths)
        filtered_count = len(self.filtered_image_paths)
        
        if total_images > 0:
            if filtered_count == total_images:
                count_text = f"{total_images} images"
            else:
                count_text = f"{filtered_count}/{total_images} images"
            
            if self.image_paths and self.current_index < len(self.image_paths):
                current_image_path = self.image_paths[self.current_index]
                if current_image_path in self.filtered_image_paths:
                    filtered_index = self.filtered_image_paths.index(current_image_path) + 1
                    count_text = f"{filtered_index}/{filtered_count} ({count_text})"
        else:
            count_text = "No images loaded"
            
        self.lbl_image_info.setText(count_text)
    
    def on_image_selected(self, item):
        """Handle image selection from the list"""
        image_path = item.data(Qt.UserRole)
        if image_path in self.image_paths:
            self.current_index = self.image_paths.index(image_path)
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
        
        # Initialize filtered paths and apply current filters
        self.filtered_image_paths = self.image_paths.copy()
        self.apply_filters()
        
        if self.image_paths:
            self.current_index = 0
            self.show_current_image()
        else:
            self.lbl_image_info.setText("No processed images found")
            self.image_label.setText("No marked images available in results folder")
        
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
            self.image_label.setText("Please process the images first using the Processing tab")
        else:
            self.lbl_image_info.setText("No images loaded")
            self.image_label.setText("No images available")
        
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
            pixmap = QPixmap(image_path)
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            ))
            
            # Update image info with filtered count
            self.update_image_list()
            
            # Highlight current image in the list
            for i in range(self.images_list.count()):
                item = self.images_list.item(i)
                item_path = item.data(Qt.UserRole)
                if item_path == image_path:
                    self.images_list.setCurrentItem(item)
                    break
            
            # Update review button state
            self.update_review_button_state()
            
            # Update question button appearances for this image
            self.update_question_button_appearance()
                
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
        is_reviewed = self.is_image_reviewed(current_image)
        
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
            
            # Update review button state and refresh filters
            self.update_review_button_state()
            self.apply_filters()  # Refresh the filtered list to reflect new status
            
        except Exception as e:
            print("Error occurs when draw stamps", e)

    def change_detected_answer(self):
        """Change the detected answer for selected question"""
        if not self.image_paths:
            return
        # Disable button to prevent multiple clicks
        self.btn_change_answer.setEnabled(False)  # Disable button during processing

        current_image = os.path.basename(self.image_paths[self.current_index])
        question_num = self.selected_question  # Use selected question from grid
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
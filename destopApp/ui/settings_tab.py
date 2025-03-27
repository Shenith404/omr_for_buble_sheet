from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QFormLayout,
    QGroupBox,  # This was missing
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox
)

class SettingsTab(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Processing Parameters
        form_group = QGroupBox("OMR Processing Settings")
        form_layout = QFormLayout()
        
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(0, 255)
        self.threshold_spin.setValue(170)
        
        self.blur_spin = QSpinBox()
        self.blur_spin.setRange(1, 15)
        self.blur_spin.setValue(5)
        
        self.canny_min = QSpinBox()
        self.canny_min.setRange(1, 100)
        self.canny_min.setValue(10)
        
        self.canny_max = QSpinBox()
        self.canny_max.setRange(1, 200)
        self.canny_max.setValue(50)
        
        form_layout.addRow("Threshold Value:", self.threshold_spin)
        form_layout.addRow("Blur Kernel Size:", self.blur_spin)
        form_layout.addRow("Canny Min Threshold:", self.canny_min)
        form_layout.addRow("Canny Max Threshold:", self.canny_max)
        form_group.setLayout(form_layout)
        
        layout.addWidget(form_group)
        layout.addStretch()
        self.setLayout(layout)
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTableWidget, 
    QTableWidgetItem, QHeaderView, QGroupBox
)
from PySide6.QtCore import Qt
class ResultsTab(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Results Table
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Question", "Answer", "Confidence"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Example data (replace with your OMR results)
        self.update_results([
            {"question": 1, "answer": "A", "confidence": 95.6},
            {"question": 2, "answer": "B", "confidence": 87.2},
        ])
        
        layout.addWidget(self.table)
        self.setLayout(layout)
    
    def update_results(self, results):
        self.table.setRowCount(len(results))
        for row, data in enumerate(results):
            self.table.setItem(row, 0, QTableWidgetItem(str(data["question"])))
            self.table.setItem(row, 1, QTableWidgetItem(data["answer"]))
            self.table.setItem(
                row, 2, 
                QTableWidgetItem(f"{data['confidence']:.1f}%")
            )
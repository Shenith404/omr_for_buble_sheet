from PySide6.QtWidgets import (QDialog, QVBoxLayout, QGridLayout, 
                               QLineEdit, QLabel, QPushButton)
from PySide6.QtGui import QIntValidator # Import the validator
from PySide6.QtCore import Qt, QTimer

from .security_manager import SecurityManager

MAX_ATTEMPTS = 3
LOCKOUT_SECONDS = 30

class PinLockDialog(QDialog):
    """A versatile PIN dialog for setting and verifying a PIN with keyboard support."""

    def __init__(self, security_manager: SecurityManager, mode: str = 'verify', parent=None):
        super().__init__(parent)
        self.setWindowTitle("PIN Lock")
        self.setModal(True)
        
        self._sm = security_manager
        self._mode = mode
        self._pin_to_confirm = ""
        self._failed_attempts = 0

        # --- UI Widgets ---
        # --- CHANGE 1: Allow keyboard input by removing readOnly=True ---
        self.pin_display = QLineEdit(
            echoMode=QLineEdit.EchoMode.Password, 
            alignment=Qt.AlignmentFlag.AlignCenter
        )
        
        # --- CHANGE 2: Restrict keyboard input to numbers only ---
        # Set a validator that allows only integers (e.g., up to 8 digits)
        self.pin_display.setValidator(QIntValidator(0, 99999999))
        
        # --- CHANGE 3: Map the keyboard's 'Enter' key to the submit action ---
        self.pin_display.returnPressed.connect(self._process_pin_entry)
        
        self.feedback_label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        
        self._setup_ui()
        self._update_ui_for_mode()
        self.pin_display.setFocus() # Set focus to the input field on start

    def _setup_ui(self):
        """Build the static parts of the UI."""
        layout = QVBoxLayout(self)
        layout.addWidget(self.feedback_label)
        layout.addWidget(self.pin_display)

        self.keypad_layout = QGridLayout()
        buttons = ['7', '8', '9', '4', '5', '6', '1', '2', '3', 'Clear', '0', 'Enter']
        positions = [(i, j) for i in range(4) for j in range(3)]
        
        for position, text in zip(positions, buttons):
            button = QPushButton(text)
            button.clicked.connect(lambda checked=False, txt=text: self._on_button_clicked(txt))
            self.keypad_layout.addWidget(button, *position)
            
        layout.addLayout(self.keypad_layout)
        self.setFixedSize(250, 320) # Slightly taller for better spacing

    def _update_ui_for_mode(self):
        """Update labels based on whether we are setting or verifying."""
        if self._mode == 'set':
            self.feedback_label.setText("Set a new PIN")
        else:
            self.feedback_label.setText("Enter your PIN")

    def _on_button_clicked(self, text: str):
        if text == "Enter":
            self._process_pin_entry()
        # --- CHANGE 4: Map the 'Clear' button to a single backspace action ---
        elif text == "Clear":
            self.pin_display.backspace()
        else:
            self.pin_display.setText(self.pin_display.text() + text)

    def _process_pin_entry(self):
        entered_pin = self.pin_display.text()
        if not entered_pin:
            return

        if self._mode == 'set':
            self._handle_set_pin(entered_pin)
        else:
            self._handle_verify_pin(entered_pin)

    def _handle_set_pin(self, pin: str):
        if not self._pin_to_confirm:
            self._pin_to_confirm = pin
            self.feedback_label.setText("Confirm your PIN")
            self.pin_display.clear()
        elif pin == self._pin_to_confirm:
            self._sm.set_pin(pin)
            self.accept()
        else:
            self.feedback_label.setText("PINs do not match. Try again.")
            self._pin_to_confirm = ""
            self.pin_display.clear()

    def _handle_verify_pin(self, pin: str):
        if self._sm.verify_pin(pin):
            self.accept()
        else:
            self._failed_attempts += 1
            self.pin_display.clear()
            if self._failed_attempts >= MAX_ATTEMPTS:
                self._lock_ui()
            else:
                remaining = MAX_ATTEMPTS - self._failed_attempts
                self.feedback_label.setText(f"Incorrect PIN. {remaining} attempts left.")

    def _lock_ui(self):
        """Lock the UI for a set duration after too many failed attempts."""
        self.feedback_label.setText(f"Too many attempts. Locked for {LOCKOUT_SECONDS}s.")
        self.keypad_layout.setEnabled(False)
        self.pin_display.setEnabled(False)
        
        self._lockout_timer = QTimer(self)
        self._lockout_timer.setSingleShot(True)
        self._lockout_timer.timeout.connect(self._unlock_ui)
        self._lockout_timer.start(LOCKOUT_SECONDS * 1000)

    def _unlock_ui(self):
        """Re-enable UI after lockout and reset attempts."""
        self._failed_attempts = 0
        self.keypad_layout.setEnabled(True)
        self.pin_display.setEnabled(True)
        self.pin_display.setFocus()
        self._update_ui_for_mode()
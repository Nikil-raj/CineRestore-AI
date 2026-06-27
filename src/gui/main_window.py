from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QLabel,
    QMainWindow,
    QPushButton,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget
)
from src.core.config import (
    APP_NAME,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.build_ui()
    def build_ui(self):
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        toolbar.addAction(QAction("Open Video", self))
        status = QStatusBar()
        self.setStatusBar(status)
        status.showMessage("Ready")
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        title = QLabel("CineRestore AI")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            font-size:30px;
            font-weight:bold;
        """)
        button = QPushButton("?? Open Video")
        layout.addStretch()
        layout.addWidget(title)
        layout.addSpacing(20)
        layout.addWidget(button)
        layout.addStretch()

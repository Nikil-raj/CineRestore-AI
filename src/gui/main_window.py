from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QLabel,
    QMainWindow,
    QPushButton,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from src.controllers.video_controller import VideoController
from src.core.config import APP_NAME, WINDOW_HEIGHT, WINDOW_WIDTH
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.controller = VideoController(self)
        self.build_ui()
    def build_ui(self):
        toolbar = QToolBar("Toolbar")
        self.addToolBar(toolbar)
        open_action = QAction("Open Video", self)
        open_action.triggered.connect(self.controller.open_video)
        toolbar.addAction(open_action)
        status = QStatusBar()
        status.showMessage("Ready")
        self.setStatusBar(status)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout()
        title = QLabel("?? CineRestore AI")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            font-size:30px;
            font-weight:bold;
        """)
        self.video_label = QLabel("No video loaded.")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setWordWrap(True)
        open_button = QPushButton("?? Open Video")
        open_button.clicked.connect(self.controller.open_video)
        layout.addStretch()
        layout.addWidget(title)
        layout.addSpacing(20)
        layout.addWidget(open_button)
        layout.addSpacing(20)
        layout.addWidget(self.video_label)
        layout.addStretch()
        central.setLayout(layout)

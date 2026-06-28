from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from src.controllers.ai_controller import AIController
from src.controllers.extraction_controller import ExtractionController
from src.controllers.video_controller import VideoController
from src.core.config import APP_NAME, WINDOW_WIDTH, WINDOW_HEIGHT


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle(APP_NAME)
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)

        self.current_video = None

        self.controller = VideoController(self)
        self.extract_controller = ExtractionController(self)
        self.ai_controller = AIController(self)

        self.build_ui()

    def build_ui(self):

        # ---------------- Toolbar ----------------

        toolbar = QToolBar("Toolbar")
        self.addToolBar(toolbar)

        open_action = QAction("Open Video", self)
        open_action.triggered.connect(
            self.controller.open_video
        )

        toolbar.addAction(open_action)

        # ---------------- Status Bar ----------------

        status = QStatusBar()
        status.showMessage("Ready")
        self.setStatusBar(status)

        # ---------------- Central Widget ----------------

        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout()
        central.setLayout(root)

        # ---------------- Title ----------------

        title = QLabel("🎬 CineRestore AI")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            """
            font-size:30px;
            font-weight:bold;
            """
        )

        root.addWidget(title)

        # ---------------- Main Layout ----------------

        content = QHBoxLayout()

        # ===== Preview =====

        preview_frame = QFrame()

        preview_layout = QVBoxLayout()

        preview_title = QLabel("Preview")
        preview_title.setAlignment(Qt.AlignCenter)

        self.preview_label = QLabel("No Preview")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(600, 350)
        self.preview_label.setFrameShape(QFrame.Box)

        preview_layout.addWidget(preview_title)
        preview_layout.addWidget(self.preview_label)

        preview_frame.setLayout(preview_layout)

        # ===== Video Info =====

        info_frame = QFrame()

        info_layout = QVBoxLayout()

        info_title = QLabel("Video Information")
        info_title.setAlignment(Qt.AlignCenter)

        self.video_label = QLabel("No video loaded.")
        self.video_label.setWordWrap(True)
        self.video_label.setAlignment(Qt.AlignTop)

        info_layout.addWidget(info_title)
        info_layout.addWidget(self.video_label)

        info_frame.setLayout(info_layout)

        # ===== Add Panels =====

        content.addWidget(preview_frame, 2)
        content.addWidget(info_frame, 1)

        root.addLayout(content)

        # ---------------- Buttons ----------------

        open_button = QPushButton("📂 Open Video")
        open_button.clicked.connect(
            self.controller.open_video
        )

        root.addWidget(open_button)

        extract_button = QPushButton("🎞 Extract Frames")
        extract_button.clicked.connect(
            self.extract_controller.extract_frames
        )

        root.addWidget(extract_button)

        enhance_button = QPushButton("✨ Enhance Frames")
        enhance_button.clicked.connect(
            self.ai_controller.enhance_frames
        )

        root.addWidget(enhance_button)

        # ---------------- Progress ----------------

        self.progress_label = QLabel("Ready")
        self.progress_label.setAlignment(Qt.AlignCenter)

        root.addWidget(self.progress_label)
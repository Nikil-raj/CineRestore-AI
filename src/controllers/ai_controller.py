from PySide6.QtCore import QThread
from PySide6.QtWidgets import QMessageBox

from src.workers.ai_worker import AIWorker


class AIController:

    def __init__(self, parent):
        self.parent = parent
        self.thread = None
        self.worker = None

    def enhance_frames(self):

        self.parent.progress_label.setText(
            "Preparing AI..."
        )

        self.thread = QThread()

        self.worker = AIWorker()

        self.worker.moveToThread(self.thread)

        self.thread.started.connect(
            self.worker.run
        )

        self.worker.progress.connect(
            self.update_progress
        )

        self.worker.finished.connect(
            self.finished
        )

        self.worker.error.connect(
            self.error
        )

        self.worker.finished.connect(
            self.thread.quit
        )

        self.worker.finished.connect(
            self.worker.deleteLater
        )

        self.thread.finished.connect(
            self.thread.deleteLater
        )

        self.thread.start()

    def update_progress(self, current, total):

        self.parent.progress_label.setText(
            f"Enhancing Frame {current} / {total}"
        )

    def finished(self):

        self.parent.progress_label.setText(
            "Enhancement Completed"
        )

        QMessageBox.information(
            self.parent,
            "Completed",
            "All frames have been processed."
        )

    def error(self, message):

        self.parent.progress_label.setText(
            "Enhancement Failed"
        )

        QMessageBox.critical(
            self.parent,
            "Error",
            message
        )
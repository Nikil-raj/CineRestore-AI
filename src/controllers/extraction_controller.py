from pathlib import Path
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QMessageBox
from src.workers.extraction_worker import ExtractionWorker
class ExtractionController:
    def __init__(self, parent):
        self.parent = parent
        self.thread = None
        self.worker = None
    def extract_frames(self):
        if not hasattr(self.parent, "current_video"):
            QMessageBox.warning(
                self.parent,
                "No Video",
                "Please load a video first."
            )
            return
        frames_folder = Path("temp/frames")
        if frames_folder.exists():
            png_count = len(list(frames_folder.glob("*.png")))
            if png_count > 0:
                reply = QMessageBox.question(
                    self.parent,
                    "Frames Already Exist",
                    f"{png_count} extracted frames already exist.\n\n"
                    "Do you want to extract them again?\n\n"
                    "Yes = Delete and Extract Again\n"
                    "No = Use Existing Frames"
                )
                if reply == QMessageBox.StandardButton.No:
                    self.parent.progress_label.setText(
                        f"Using existing {png_count} frames."
                    )
                    return
                for file in frames_folder.glob("*.png"):
                    file.unlink()
        self.parent.progress_label.setText("Extracting... 0%")
        self.thread = QThread()
        self.worker = ExtractionWorker(
            self.parent.current_video
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.finished)
        self.worker.error.connect(self.show_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.cleanup)
        self.thread.start()
    def cleanup(self):
        self.thread = None
        self.worker = None
    def update_progress(self, percent):
        self.parent.progress_label.setText(
            f"Extracting... {percent}%"
        )
    def finished(self, total):
        self.parent.progress_label.setText(
            f"Finished! {total} frames extracted."
        )
        QMessageBox.information(
            self.parent,
            "Completed",
            f"{total} frames extracted successfully."
        )
    def show_error(self, message):
        QMessageBox.critical(
            self.parent,
            "Error",
            message
        )

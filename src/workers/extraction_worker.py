from PySide6.QtCore import QObject, Signal
from src.video.frame_extractor import FrameExtractor
class ExtractionWorker(QObject):
    progress = Signal(int)
    finished = Signal(int)
    error = Signal(str)
    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
    def run(self):
        try:
            total = FrameExtractor.extract(
                self.video_path,
                self.progress.emit
            )
            self.finished.emit(total)
        except Exception as e:
            self.error.emit(str(e))

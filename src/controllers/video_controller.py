from pathlib import Path
from PySide6.QtWidgets import QFileDialog
class VideoController:
    def __init__(self, parent):
        self.parent = parent
    def open_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.parent,
            "Open Video",
            "",
            "Video Files (*.mp4 *.mkv *.avi *.mov)"
        )
        if not file_path:
            self.parent.statusBar().showMessage("No video selected.")
            return
        self.parent.video_label.setText(
            f"Selected Video:\n\n{Path(file_path).name}\n\n{file_path}"
        )
        self.parent.statusBar().showMessage("Video Loaded Successfully")

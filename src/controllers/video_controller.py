from pathlib import Path

import cv2

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QFileDialog, QMessageBox

from src.services.video_service import VideoService
from src.video.video_loader import VideoLoader


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
            self.parent.statusBar().showMessage(
                "No video selected."
            )
            return

        # Store current video for future processing
        self.parent.current_video = file_path

        try:

            metadata = VideoService.load_video(file_path)

            frame = VideoLoader.load_preview_frame(file_path)

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            height, width, channel = frame.shape

            image = QImage(
                frame.data,
                width,
                height,
                channel * width,
                QImage.Format_RGB888
            )

            pixmap = QPixmap.fromImage(image)

            pixmap = pixmap.scaled(
                self.parent.preview_label.width(),
                self.parent.preview_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            self.parent.preview_label.setPixmap(
                pixmap
            )

            duration = metadata["duration"]

            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = int(duration % 60)

            self.parent.video_label.setText(
                f"""Movie

{Path(file_path).name}

Resolution : {metadata["width"]} x {metadata["height"]}
FPS        : {metadata["fps"]}
Frames     : {metadata["frames"]}
Duration   : {hours:02}:{minutes:02}:{seconds:02}
"""
            )

            self.parent.statusBar().showMessage(
                "Video Loaded Successfully"
            )

        except Exception as e:

            QMessageBox.critical(
                self.parent,
                "Error",
                str(e)
            )
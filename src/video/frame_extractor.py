import cv2
from pathlib import Path
class FrameExtractor:
    @staticmethod
    def extract(video_path: str, progress_callback=None):
        output = Path("temp/frames")
        output.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Unable to open video.")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            filename = output / f"frame_{frame_number:06}.png"
            cv2.imwrite(str(filename), frame)
            frame_number += 1
            if progress_callback and total_frames > 0:
                percent = int((frame_number / total_frames) * 100)
                progress_callback(percent)
        cap.release()
        return frame_number
